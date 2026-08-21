"""JoyLTX_LoadModel: one loader for both model families (same idea as the H3 pack's loader).

- Lists every file under models/diffusion_models (+ models/unet): core's list plus a recursive walk for
  .gguf (core never lists .gguf, and files under diffusion_models/gguf/ would be missed).
- .gguf routes through ComfyUI-GGUF's Unet Loader (clear error if that pack is missing); anything else
  through the stock Load Diffusion Model.
- VALIDATE_INPUTS accepts any name and load() resolves a moved file by UNIQUE basename, so a canvas saved
  with the bare public file name still queues when the user filed the model in a subfolder.
- Warns once when only comfy-kitchen's eager backend is live (comfy-native quantised files run ~2x slower
  there; GGUF is unaffected).
"""
import os
import folder_paths
import nodes


def _list_names(folder="diffusion_models"):
    try:
        files = list(folder_paths.get_filename_list(folder))
    except Exception:
        files = []
    gguf = []
    try:
        dirs = folder_paths.get_folder_paths(folder)
    except Exception:
        dirs = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, fs in os.walk(d, followlinks=True):
            for f in fs:
                if f.lower().endswith(".gguf"):
                    gguf.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(set(files) | set(gguf))


class JoyLTX_LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        names = _list_names("diffusion_models") or ["(no models found)"]
        return {"required": {
            "model_name": (names, {"tooltip": "Any file in models/diffusion_models (or models/unet): "
                                            ".safetensors loads with the stock loader, .gguf with ComfyUI-GGUF's "
                                            "(install ComfyUI-GGUF for that). RTX 30/40: take the GGUF; RTX 50: comfy-native. "
                                            "A saved name that moved into a subfolder is resolved by file name."}),
        }, "optional": {
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"tooltip": "Only used for .safetensors."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Load Diffusion Model that also takes .gguf files (via ComfyUI-GGUF), resolves moved files by name."

    @classmethod
    def VALIDATE_INPUTS(cls, model_name=None, **kwargs):
        return True  # resolved (and clearly reported) in load()

    @staticmethod
    def _resolve(model_name):
        want = os.path.basename(str(model_name)).lower()
        cands = []
        for folder in ("diffusion_models", "unet"):
            for f in _list_names(folder):
                if os.path.basename(f).lower() == want:
                    cands.append(f)
        if model_name in cands:
            return model_name
        uniq = sorted(set(cands))
        if len(uniq) == 1:
            print("[JoyLTX] %r is not at the saved path; resolved by file name to %r" % (model_name, uniq[0]), flush=True)
            return uniq[0]
        if len(uniq) > 1:
            raise RuntimeError("[JoyLTX] %r matches several files (%s) - pick one in the dropdown." % (model_name, ", ".join(uniq)))
        roots = folder_paths.get_folder_paths("diffusion_models")
        raise RuntimeError("[JoyLTX] model file not found: %r. Searched: %s. Pick your file in this node's dropdown."
                           % (model_name, ", ".join(roots)))

    @staticmethod
    def _warn_quant_backend(model_name):
        if str(model_name).lower().endswith(".gguf"):
            return
        try:
            import comfy_kitchen as ck
            b = ck.list_backends()
        except Exception:
            return
        def live(name):
            info = b.get(name) if isinstance(b, dict) else None
            return bool(info) and info.get("available") and not info.get("disabled")
        if live("cuda") or live("triton"):
            return
        print("[JoyLTX] NOTE: only comfy-kitchen's 'eager' backend is live here: comfy-native quantised files "
              "(int8/w4a8/nvfp4/fp8) dequantise every step (~2x slower). Add --enable-triton-backend to the launch "
              "line, or use the GGUF build.", flush=True)

    def load(self, model_name, weight_dtype="default"):
        model_name = self._resolve(model_name)
        self._warn_quant_backend(model_name)
        if model_name.lower().endswith(".gguf"):
            cls = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
            if cls is None:
                raise RuntimeError("[JoyLTX] '%s' is a GGUF: install ComfyUI-GGUF (Manager -> 'ComfyUI-GGUF') "
                                   "or pick a .safetensors file." % os.path.basename(model_name))
            out = cls().load_unet(model_name)
        else:
            out = nodes.UNETLoader().load_unet(model_name, weight_dtype)
        try:
            out[0].joyltx_model_name = model_name    # lets the LoRA stack warn on cross-generation audio LoRAs
        except Exception:
            pass
        return out


NODE_CLASS_MAPPINGS = {"JoyLTX_LoadModel": JoyLTX_LoadModel}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_LoadModel": "JoyLTX Load Model"}
