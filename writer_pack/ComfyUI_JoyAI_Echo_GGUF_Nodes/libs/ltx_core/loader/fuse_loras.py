import torch

from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.quantization.fp8_cast import calculate_weight_float8
from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    destination_sd: StateDict | None = None,
) -> StateDict:
    lora_sd_and_strengths = _normalize_kohya_flat_keys(lora_sd_and_strengths, model_sd)

    sd = {}
    if destination_sd is not None:
        sd = destination_sd.sd
    size = 0
    device = torch.device("meta")
    inner_dtypes = set()
    fused_count = 0
    for key, weight in model_sd.sd.items():
        if weight is None:
            continue
        # Skip scale keys - they are handled together with their weight keys
        if key.endswith(".weight_scale"):
            continue
        device = weight.device
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

        scale_key = key.replace(".weight", ".weight_scale") if key.endswith(".weight") else None
        is_scaled_fp8 = scale_key is not None and scale_key in model_sd.sd

        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, device)
        if deltas is not None:
            fused_count += 1
        fused = _fuse_deltas(deltas, weight, key, sd, target_dtype, device, is_scaled_fp8, scale_key, model_sd)

        sd.update(fused)
        for tensor in fused.values():
            inner_dtypes.add(tensor.dtype)
            size += tensor.nbytes

    if lora_sd_and_strengths:
        if fused_count == 0:
            print("[LoRA fuse] WARNING: a LoRA was provided but ZERO keys matched the "
                  "model - the LoRA had NO effect (key-naming mismatch?).", flush=True)
        else:
            print(f"[LoRA fuse] {fused_count} weight(s) fused.", flush=True)

    if destination_sd is not None:
        return destination_sd
    return StateDict(sd, device, size, inner_dtypes)


class _RenamedLoraSD:
    """Minimal .sd holder for a key-normalized LoRA state dict."""

    __slots__ = ("sd",)

    def __init__(self, sd):
        self.sd = sd


def _normalize_kohya_flat_keys(lora_sd_and_strengths, model_sd):
    """Translate kohya-FLATTENED LoRA keys onto the model's dotted paths.

    Kohya exports name modules as 'lora_unet_transformer_blocks_0_attn1_to_q'
    (prefix + every '.' flattened to '_'). Un-flattening by string rules is
    ambiguous ('to_out_0' -> 'to_out.0' but 'audio_attn1' keeps its '_'), so
    instead we flatten every MODEL weight prefix the same way and look each
    LoRA stem up exactly. Non-kohya keys pass through untouched.
    """
    flat_to_dotted = {}
    for mk in model_sd.sd:
        if mk.endswith(".weight"):
            pre = mk[: -len(".weight")]
            flat_to_dotted[pre.replace(".", "_")] = pre

    out = []
    for lsd, coef in lora_sd_and_strengths:
        src = lsd.sd
        if not any(k.startswith("lora_unet_") for k in src):
            out.append((lsd, coef))
            continue
        new, mapped = {}, 0
        for k, v in src.items():
            stem, dot, suffix = k.partition(".")
            if dot and stem.startswith("lora_unet_"):
                dotted = flat_to_dotted.get(stem[len("lora_unet_"):])
                if dotted is not None:
                    new[f"{dotted}.{suffix}"] = v
                    mapped += 1
                    continue
            new[k] = v
        print(f"[LoRA fuse] kohya-flattened LoRA detected: {mapped}/{len(src)} keys "
              f"mapped onto model paths.", flush=True)
        out.append((_RenamedLoraSD(new), coef))
    return out


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    # Support both PEFT (lora_A/lora_B) and kohya (lora_down/lora_up) naming,
    # with kohya-standard alpha/rank scaling when an .alpha scalar is present
    # (a missing alpha, or alpha == rank, leaves the strength unchanged).
    variants = (
        (f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"),
        (f"{prefix}.lora_down.weight", f"{prefix}.lora_up.weight"),
    )
    alpha_key = f"{prefix}.alpha"
    for lsd, coef in lora_sd_and_strengths:
        for key_a, key_b in variants:
            if key_a not in lsd.sd or key_b not in lsd.sd:
                continue
            a = lsd.sd[key_a].to(device=device)
            b = lsd.sd[key_b].to(device=device)
            scale = coef
            alpha = lsd.sd.get(alpha_key)
            if alpha is not None and a.shape[0]:
                scale = coef * float(alpha) / a.shape[0]
            product = torch.matmul(b * scale, a)
            del a, b
            deltas.append(product.to(dtype=dtype))
            break
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    return torch.sum(torch.stack(deltas, dim=0), dim=0)


def _fuse_deltas(
    deltas: torch.Tensor | None,
    weight: torch.Tensor,
    key: str,
    sd: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    device: torch.device,
    is_scaled_fp8: bool,
    scale_key: str | None,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    if deltas is None:
        if key in sd:
            return {}
        fused = _copy_weight_without_lora(weight, key, target_dtype, device, is_scaled_fp8, scale_key, model_sd)
    elif weight.dtype == torch.float8_e4m3fn:
        if is_scaled_fp8:
            fused = _fuse_delta_with_scaled_fp8(deltas, weight, key, scale_key, model_sd)
        else:
            fused = _fuse_delta_with_cast_fp8(deltas, weight, key, target_dtype, device)
    elif weight.dtype == torch.bfloat16:
        fused = _fuse_delta_with_bfloat16(deltas, weight, key, target_dtype)
    else:
        raise ValueError(f"Unsupported dtype: {weight.dtype}")

    return fused


def _copy_weight_without_lora(
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    device: torch.device,
    is_scaled_fp8: bool,
    scale_key: str | None,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Copy original weight (and scale if applicable) when no LoRA affects this key."""
    result = {key: weight.clone().to(dtype=target_dtype, device=device)}
    if is_scaled_fp8:
        result[scale_key] = model_sd.sd[scale_key].clone()
    return result


def _fuse_delta_with_scaled_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    scale_key: str,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize scaled FP8 weight, add LoRA delta, and re-quantize."""
    weight_scale = model_sd.sd[scale_key]

    original_weight = weight.t().to(torch.float32) * weight_scale

    new_weight = original_weight + deltas.to(torch.float32)

    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
    return {key: new_fp8_weight, scale_key: new_weight_scale}


def _fuse_delta_with_cast_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with cast-only FP8 weight (no scale factor)."""
    if str(device).startswith("cuda"):
        deltas = calculate_weight_float8(deltas, weight)
    else:
        deltas.add_(weight.to(dtype=deltas.dtype, device=device))
    return {key: deltas.to(dtype=target_dtype)}


def _fuse_delta_with_bfloat16(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with bfloat16 weight."""
    deltas.add_(weight)
    return {key: deltas.to(dtype=target_dtype)}
