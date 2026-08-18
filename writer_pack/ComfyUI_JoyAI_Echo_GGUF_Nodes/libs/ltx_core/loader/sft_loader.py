import json

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps

# ---- ComfyUI-quantized (comfy_quant) INT8 ConvRot support (Rebels patch) ----
# The published INT8 ConvRot surgical builds store, per quantized linear:
#   <layer>.weight        int8   [out, in]   (input dim rotated in 256-blocks)
#   <layer>.weight_scale  fp32   [out, 1]    (symmetric per-output-channel)
#   <layer>.comfy_quant   uint8  JSON marker {"format": "int8_tensorwise",
#                                             "convrot": true, "convrot_groupsize": 256}
# This loader reconstructs bf16 weights at load: dequant = int8 * scale, then
# un-rotate. The export rotated with W_rot = grouped(W) @ H.T where H is the
# NORMALIZED REGULAR Hadamard (Kronecker powers of the regular H4 below,
# / sqrt(size)) - symmetric AND orthogonal, so H is its own inverse and the
# reconstruction is simply grouped(W_rot) @ H. Matches comfy-quants
# formats/convrot.py bit-for-bit (regular H4, no all-ones column - this is NOT
# the Sylvester Hadamard; using scipy.linalg.hadamard here would corrupt every
# rotated layer).

_HADAMARD_CACHE = {}


def _regular_hadamard(size, device, dtype):
    key = (size, str(device), dtype)
    h = _HADAMARD_CACHE.get(key)
    if h is not None:
        return h
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h
    return h


def _dequant_comfy_int8(w_int8, scale, marker, name):
    conf = json.loads(bytes(marker.numpy().tobytes()).decode("utf-8"))
    fmt = conf.get("format")
    if fmt != "int8_tensorwise":
        raise ValueError(
            f"{name}: comfy_quant format '{fmt}' is not supported by this "
            "loader (only int8_tensorwise, e.g. the INT8 ConvRot builds). "
            "Use a bf16 checkpoint or a GGUF instead.")
    # dequant + un-rotate in fp32 on the GPU when one is free (a 22B DiT is
    # ~1500 of these; CPU works but adds minutes), emit bf16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        w = w_int8.to(device=dev, dtype=torch.float32) * scale.to(
            device=dev, dtype=torch.float32)
        if conf.get("convrot"):
            g = int(conf.get("convrot_groupsize", 256))
            out_f, in_f = w.shape
            if in_f % g:
                raise ValueError(f"{name}: in_features {in_f} not divisible by "
                                 f"convrot_groupsize {g}")
            h = _regular_hadamard(g, w.device, torch.float32)
            w = torch.matmul(w.reshape(out_f, in_f // g, g), h).reshape(out_f, in_f)
        return w.to(device="cpu", dtype=torch.bfloat16)
    except torch.cuda.OutOfMemoryError:
        # retry the whole thing on CPU rather than dying mid-load
        w = w_int8.to(dtype=torch.float32) * scale.to(dtype=torch.float32)
        if conf.get("convrot"):
            g = int(conf.get("convrot_groupsize", 256))
            out_f, in_f = w.shape
            h = _regular_hadamard(g, "cpu", torch.float32)
            w = torch.matmul(w.reshape(out_f, in_f // g, g), h).reshape(out_f, in_f)
        return w.to(torch.bfloat16)


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                # comfy_quant awareness: int8_tensorwise markers identify layers
                # we reconstruct to bf16 at load (marker + scale sidecars are
                # then never emitted). Any OTHER comfy_quant format (e.g. the
                # float8_e4m3fn fp8mixed gemma builds) passes through RAW -
                # weight, weight_scale and marker intact - because a downstream
                # consumer owns it (_swap_gemma_fp8 in rebels_loaders). That is
                # the pre-int8-patch behavior those files were built against.
                cq_bases = {k[: -len(".comfy_quant")]
                            for k in safetensor_keys if k.endswith(".comfy_quant")}
                int8_bases = set()
                for b in cq_bases:
                    try:
                        conf = json.loads(bytes(
                            f.get_tensor(b + ".comfy_quant").numpy().tobytes()
                        ).decode("utf-8"))
                    except Exception:
                        conf = {}
                    if conf.get("format") == "int8_tensorwise":
                        int8_bases.add(b)
                if int8_bases:
                    print(f"[JoyEcho] comfy_quant checkpoint: reconstructing "
                          f"{len(int8_bases)} INT8 layers to bf16 at load "
                          f"(ConvRot un-rotation included)...", flush=True)
                if len(cq_bases) - len(int8_bases) > 0:
                    print(f"[JoyEcho] comfy_quant checkpoint: passing "
                          f"{len(cq_bases) - len(int8_bases)} non-int8 quantized "
                          f"layers through untouched (handled downstream, e.g. "
                          f"fp8mixed gemma).", flush=True)
                for name in safetensor_keys:
                    if int8_bases:
                        if name.endswith((".comfy_quant", ".weight_scale")) and \
                                name.rsplit(".", 1)[0] in int8_bases:
                            continue
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    # copy=False can return a tensor that aliases the safe_open
                    # mmap; the mmap is unmapped when this `with` block exits, so
                    # tensors kept alive by later builders (e.g. all the VAE
                    # encoders/decoders held at once in create_vae_wrappers) dangle
                    # and the next load faults with a native access violation on
                    # Windows. Force an owned copy off the mmap. (Rebels local patch)
                    if int8_bases and name.endswith(".weight") and \
                            name[: -len(".weight")] in int8_bases:
                        base = name[: -len(".weight")]
                        value = _dequant_comfy_int8(
                            f.get_tensor(name),
                            f.get_tensor(base + ".weight_scale"),
                            f.get_tensor(base + ".comfy_quant"),
                            name).to(device=device)
                    else:
                        value = f.get_tensor(name).to(device=device, non_blocking=True, copy=True)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta is None or "config" not in meta:
                return {}
            return json.loads(meta["config"])

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)
