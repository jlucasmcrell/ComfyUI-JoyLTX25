"""Torch-native FP8 scaled-matmul inference (no tensorrt_llm dependency).

Weights are stored float8_e4m3fn (plain cast, scale 1.0 - bf16 transformer
weights sit well inside the E4M3 range) and matmuls execute IN fp8 via
torch._scaled_mm on hardware with native fp8 tensor cores (sm_89+; RTX 5090
sm_120 runs it at ~2x bf16 throughput). Inputs are quantized dynamically
per-tensor each call. This removes BOTH costs of the fp8_cast path: the
per-layer bf16 upcast tax AND (with the weights resident) the sequential-
offload PCIe streaming.
"""

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.quantization.fp8_cast import _replace_fwd_with_upcast

_F8_MAX = 448.0  # float8_e4m3fn finite max

_SM_OK: dict = {}  # device -> bool: native fp8 mm available (sm_89+)


def _scaled_mm_supported(device) -> bool:
    ok = _SM_OK.get(device)
    if ok is None:
        try:
            ok = torch.cuda.get_device_capability(device) >= (8, 9)
        except Exception:
            ok = False
        _SM_OK[device] = ok
        if not ok:
            print(f"[ltx_core] fp8 scaled-mm: {device} lacks sm_89+ fp8 cores; "
                  "falling back to per-layer upcast on this device.", flush=True)
    return ok


def _replace_fwd_with_scaled_mm(layer: torch.nn.Linear) -> None:
    """Swap layer.forward for a runtime-branching fp8 path.

    IMPORTANT: this runs on the META model, BEFORE weights load - so the
    branch on weight dtype must happen at CALL time, never at swap time
    (a swap-time dtype check sees only meta/bf16 placeholders and matches
    nothing, silently disabling the whole mode).

    At call time: fp8 weight + CUDA + 16-aligned dims -> torch._scaled_mm
    (plain-cast weights, dequant scale 1.0); fp8 otherwise -> per-layer
    upcast; non-fp8 weight -> the original forward untouched.
    """
    layer.original_forward = layer.forward
    # per-device cached constant scale for the (plain-cast) weight
    _scale_w: dict = {}

    def new_forward(*args, **_kwargs) -> torch.Tensor:
        x = args[0]
        w = layer.weight
        if w.dtype != torch.float8_e4m3fn:
            return layer.original_forward(*args, **_kwargs)
        if layer.bias is not None and layer.bias.dtype == torch.float8_e4m3fn:
            layer.bias.data = layer.bias.data.to(torch.bfloat16)  # one-time restore
        if (not (x.is_cuda and w.is_cuda)) or w.shape[0] % 16 or w.shape[1] % 16 \
                or not _scaled_mm_supported(x.device):
            # CPU pass, non-16-aligned dims, or pre-Ada GPU: per-layer upcast math.
            w_up = w.to(x.dtype)
            b = layer.bias.to(x.dtype) if layer.bias is not None else None
            return torch.nn.functional.linear(x, w_up, b)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        # dynamic per-tensor input quantization, kept in the input dtype end to
        # end (an fp32 round-trip here costs more than the fp8 mm saves; the
        # fp8 cast saturates, so no explicit clamp is needed). Measured on the
        # 5090 at M=28672: raw fp8 mm x2.80 vs bf16, x1.52 including this quant.
        amax = x2.abs().amax().clamp_min(1e-12).float()
        q = _F8_MAX / amax
        x_q = (x2 * q.to(x2.dtype)).to(torch.float8_e4m3fn)
        dev = x.device
        sw = _scale_w.get(dev)
        if sw is None:
            sw = torch.ones((), device=dev, dtype=torch.float32)
            _scale_w[dev] = sw
        bias = layer.bias
        if bias is not None and bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        out = torch._scaled_mm(
            x_q,
            w.t(),                      # (K, N) column-major view of the (N, K) weight
            scale_a=(1.0 / q).reshape(()),
            scale_b=sw,
            bias=bias,
            out_dtype=torch.bfloat16,
        )
        return out.reshape(*orig_shape[:-1], w.shape[0]).to(x.dtype)

    layer.forward = new_forward


def _amend_forward_with_scaled_mm(model: torch.nn.Module) -> torch.nn.Module:
    """Patch EVERY Linear with the runtime-branching scaled-mm forward.

    Runs pre-load on the meta model, so no dtype filtering here - the patched
    forward decides per call: fp8 weights take the fp8 path, everything else
    runs its original forward unchanged.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            _replace_fwd_with_scaled_mm(m)
            n += 1
    print(f"[ltx_core] fp8 scaled-mm: {n} linears patched "
          f"(fp8-vs-bf16 branch decided at call time).", flush=True)
    return model


TORCH_SCALED_MM_DURING_INFERENCE = ModuleOps(
    name="torch_scaled_mm_fp8_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=_amend_forward_with_scaled_mm,
)
