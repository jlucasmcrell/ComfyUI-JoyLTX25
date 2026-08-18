"""JoyLTX_LoadModel: one loader for both model families.

Lists every file under models/diffusion_models (and models/unet) - .safetensors AND .gguf - and
dispatches: a .gguf goes through ComfyUI-GGUF's Unet Loader (GGUF) when that pack is installed,
anything else through the stock Load Diffusion Model. One canvas serves RTX 30/40 (GGUF) and
RTX 50 (comfy-native) users without swapping nodes.
"""
import os
import folder_paths
import nodes


def _names():
    seen, out = set(), []
    for kind in ("diffusion_models", "unet_gguf"):
        try:
            for n in folder_paths.get_filename_list(kind):
                if n not in seen:
                    seen.add(n); out.append(n)
        except Exception:
            pass
    return sorted(out) or ["(no models found)"]


class JoyLTX_LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (_names(), {"tooltip": "Any file in models/diffusion_models or models/unet: "
                                                ".safetensors loads with the stock loader, .gguf with ComfyUI-GGUF's "
                                                "loader (install ComfyUI-GGUF for that; RTX 30/40 = take the GGUF, RTX 50 = comfy-native)."}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"tooltip": "Only used for .safetensors."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Load Diffusion Model that also takes .gguf files (via ComfyUI-GGUF)."

    @classmethod
    def VALIDATE_INPUTS(cls, model_name, weight_dtype):
        return True  # the list is rebuilt per call; a stale name errors clearly in load()

    def load(self, model_name, weight_dtype="default"):
        if model_name.lower().endswith(".gguf"):
            cls = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
            if cls is None:
                raise RuntimeError("[JoyLTX] '%s' is a GGUF: install ComfyUI-GGUF (Manager -> 'ComfyUI-GGUF') "
                                   "or pick a .safetensors file." % os.path.basename(model_name))
            return cls().load_unet(model_name)
        return nodes.UNETLoader().load_unet(model_name, weight_dtype)


NODE_CLASS_MAPPINGS = {"JoyLTX_LoadModel": JoyLTX_LoadModel}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_LoadModel": "JoyLTX Load Model (safetensors or GGUF)"}
