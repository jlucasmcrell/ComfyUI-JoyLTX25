"""Runtime LoRA for the GGUF DiT path - fixes "lora_path is ignored on GGUF".

Attribution: the key-naming conventions, PEFT/kohya variants and alpha/rank
scaling here deliberately mirror RealRebelAI's
`libs/ltx_core/loader/fuse_loras.py` (`_prepare_deltas`,
`_normalize_kohya_flat_keys`) so the fused safetensors path and this runtime
path can never disagree about what a LoRA file means. That upstream code is
theirs; this module is ours and lives only in our patch repo.

WHY THIS EXISTS
    JoyEcho's only LoRA mechanism was fusion at load (W' = W + s*B*A), which is
    impossible on a GGUF DiT: weights are packed quantized blocks, memory-mapped
    from disk, dequantized per layer inside GGUFLinear.forward. Fusing would mean
    dequantize -> add -> REQUANTIZE (an owned RAM copy of the whole model, and a
    rank-32 delta can vanish below the quantization step). So the loader dropped
    every LoRA behind one console warning, and any GGUF render silently ran
    LoRA-free no matter what the stacker showed.

WHAT IT DOES INSTEAD
    Leaves the GGUF weights alone and adds each LoRA at runtime via forward
    hooks:  y = module(x) + scale * (x @ A^T) @ B^T
    On a quantized base this is HIGHER fidelity than fuse-and-requantize: the
    delta stays bf16 instead of being crushed into the quant grid. Cost is two
    small matmuls per hooked linear (rank 32 vs hidden 4096, ~1-2% of layer
    FLOPs).

SCALING, STATED EXPLICITLY
    Our LoRAs carry NO `.alpha` keys (verified: 0 alpha keys in
    ltx23_surfaces_v2_rank32). With no alpha, scale == strength verbatim, which
    is the PEFT convention and matches fuse_loras. If you ever train one WITH
    alpha, scale becomes strength * alpha / rank automatically.

MEMORY - READ THIS IF YOU RUN sequential_offload
    Hooks cache their A/B on the compute device on first call and keep them
    there. Measured on ltx23_surfaces_v2_rank32: 1152 A/B tensors, 176.3M
    elements, **353 MB in bf16 per LoRA** - so a three-LoRA stack pins about
    1.06 GB on the GPU, and it is NEVER streamed by the offloader. On a 32 GB
    card that is noise. On a 12 GB card running sequential_offload (which gets
    the DiT itself down to 2-3 GB) it is roughly a third of your headroom.
    Budget for it, or run fewer LoRAs, or use JOYECHO_GGUF_LORA=0.

SAFETY
    - Attach happens AFTER _materialize_meta/_rebind_swapped so hooks land on
      the final module objects.
    - A/B are plain attributes on the hook closure, never registered - so
      state_dict, .to() sweeps and the offloader never see them (same reason
      GGUFLinear keeps _qweight unregistered).
    - Kill switch: JOYECHO_GGUF_LORA=0 restores the old drop-behaviour.
    - Attach reports matched/total pairs AND the audio-touching count.
    - Hooks now prove they RAN, not just that they attached - see below.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch

__all__ = ["attach_runtime_loras", "runtime_lora_enabled", "runtime_lora_stats"]

_LORA_SUFFIX_RE = re.compile(
    r"\.(lora_A|lora_B|lora_down|lora_up)\.weight$|\.alpha$")

# Live hook registry so callers can ask "did these actually fire?" - attach
# succeeding proves nothing if something later rebuilds the modules (a second
# load, the hires second pass) and silently drops the hooks. Previously the
# console would still read "576/576 attached" while nothing ran.
_HOOKS: list = []


def runtime_lora_enabled() -> bool:
    return os.environ.get("JOYECHO_GGUF_LORA", "1").strip() not in ("0", "off", "false")


def runtime_lora_stats() -> dict:
    """{'hooks': n, 'fired': n_that_ran, 'calls': total_forward_calls}."""
    return {"hooks": len(_HOOKS),
            "fired": sum(1 for h in _HOOKS if h.calls),
            "calls": sum(h.calls for h in _HOOKS)}


def _load_lora_sd(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(path)


def _strip_prefix(key: str) -> str:
    return key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key


def _unflatten_kohya(sd: dict, module_paths) -> dict:
    if not any(k.startswith("lora_unet_") for k in sd):
        return sd
    flat_to_dotted = {p.replace(".", "_"): p for p in module_paths}
    out, mapped = {}, 0
    for k, v in sd.items():
        stem, dot, suffix = k.partition(".")
        if dot and stem.startswith("lora_unet_"):
            dotted = flat_to_dotted.get(stem[len("lora_unet_"):])
            if dotted is not None:
                out[f"{dotted}.{suffix}"] = v
                mapped += 1
                continue
        out[k] = v
    print(f"[JoyEcho GGUF-LoRA] kohya-flattened LoRA: {mapped}/{len(sd)} keys "
          f"mapped onto module paths.", flush=True)
    return out


def _collect_pairs(sd: dict, strength: float) -> dict:
    pairs = {}
    prefixes = set()
    for k in sd:
        m = _LORA_SUFFIX_RE.search(k)
        if m and not k.endswith(".alpha"):
            prefixes.add(k[: m.start()])
    for prefix in prefixes:
        for key_a, key_b in ((f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"),
                             (f"{prefix}.lora_down.weight", f"{prefix}.lora_up.weight")):
            if key_a in sd and key_b in sd:
                a, b = sd[key_a], sd[key_b]
                scale = strength
                alpha = sd.get(f"{prefix}.alpha")
                if alpha is not None and a.shape[0]:
                    scale = strength * float(alpha) / a.shape[0]
                pairs[prefix] = (a, b, scale)
                break
    return pairs


class _LoraHook:
    """Adds sum_i scale_i * (x @ A_i^T) @ B_i^T to a module's output."""

    __slots__ = ("terms", "calls")

    def __init__(self):
        self.terms = []
        self.calls = 0

    def add(self, a: torch.Tensor, b: torch.Tensor, scale: float, dtype):
        self.terms.append([a.t().contiguous().to(dtype),
                           b.t().contiguous().to(dtype), float(scale)])

    def __call__(self, module, args, output):
        self.calls += 1
        if not args:
            return output
        x = args[0]
        for term in self.terms:
            at, bt, scale = term
            if not scale:
                continue                     # strength 0 attaches but costs nothing
            if at.device != x.device or at.dtype != x.dtype:
                term[0] = at = at.to(device=x.device, dtype=x.dtype)
                term[1] = bt = bt.to(device=x.device, dtype=x.dtype)
            output = output + (x @ at) @ bt * scale
        return output


def attach_runtime_loras(root: torch.nn.Module, lora_entries,
                         compute_dtype=torch.bfloat16,
                         attach_zero: bool = False) -> int:
    """Attach runtime-LoRA hooks for [(path, strength), ...] onto `root`.

    attach_zero=True attaches strength-0 LoRAs with scale 0 instead of skipping
    them. Production wants the skip (why spend matmuls on a zero term), but the
    validation harness needs the attach: "strength 0 must equal no-LoRA" only
    tests the hook MATH if a hook is actually present. With the skip it merely
    tests the skip branch, which is why that check looked reassuring and proved
    nothing.

    Returns the number of hooked modules. Never raises for a bad LoRA file.
    """
    if not lora_entries:
        return 0
    if not runtime_lora_enabled():
        print("[JoyEcho GGUF-LoRA] disabled via JOYECHO_GGUF_LORA=0 - LoRAs "
              "NOT applied.", flush=True)
        return 0

    modules = dict(root.named_modules())
    # Suffix index, actually used this time. LoRA paths are relative to the bare
    # LTXModel; the live tree prefixes them (wrapper.model....). Precomputing
    # every suffix once turns resolve() from a full scan per pair (O(pairs x
    # modules) - millions of string compares for a 576-pair LoRA) into a dict
    # hit. Only unambiguous matches are accepted.
    by_suffix: dict = {}
    for full in modules:
        parts = full.split(".")
        for i in range(len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(full)

    def resolve(target: str):
        if target in modules:
            return target
        hits = by_suffix.get(target)
        return hits[0] if hits and len(hits) == 1 else None

    hooks: dict = {}
    for path, strength in lora_entries:
        if not strength and not attach_zero:
            print(f"[JoyEcho GGUF-LoRA] {Path(path).name}: strength 0 - skipped.",
                  flush=True)
            continue
        try:
            sd = _load_lora_sd(path)
        except Exception as e:
            print(f"[JoyEcho GGUF-LoRA] WARNING: could not read {path} "
                  f"({type(e).__name__}: {e}) - skipped.", flush=True)
            continue
        sd = {_strip_prefix(k): v for k, v in sd.items()}
        sd = _unflatten_kohya(sd, modules.keys())
        pairs = _collect_pairs(sd, strength)
        matched = missed = audio = 0
        for prefix, (a, b, scale) in pairs.items():
            target = resolve(prefix)
            mod = modules.get(target) if target else None
            if mod is None:
                missed += 1
                continue
            if "audio" in prefix.lower():
                audio += 1
            hook = hooks.get(target)
            if hook is None:
                hook = hooks[target] = _LoraHook()
                mod.register_forward_hook(hook)
                _HOOKS.append(hook)
            hook.add(a, b, scale, compute_dtype)
            matched += 1
        tag = Path(path).name
        if matched == 0:
            print(f"[JoyEcho GGUF-LoRA] WARNING: {tag}: ZERO of {len(pairs)} "
                  f"pairs matched the model - this LoRA has NO effect "
                  f"(key-naming mismatch?).", flush=True)
        else:
            print(f"[JoyEcho GGUF-LoRA] {tag}@{strength}: {matched}/{len(pairs)} "
                  f"pairs attached ({audio} audio-touching)"
                  + (f", {missed} unmatched" if missed else "") + ".", flush=True)
    if hooks:
        mb = sum(t[0].numel() + t[1].numel()
                 for h in hooks.values() for t in h.terms) * 2 / 1e6
        print(f"[JoyEcho GGUF-LoRA] runtime LoRA ACTIVE on {len(hooks)} modules "
              f"(~{mb:.0f} MB of A/B will pin to the compute device; GGUF weights "
              f"untouched; disable with JOYECHO_GGUF_LORA=0).", flush=True)
    return len(hooks)
