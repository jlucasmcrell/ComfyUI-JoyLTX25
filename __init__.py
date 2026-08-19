"""ComfyUI-JoyLTX25: Joy-LTX 2.5 (JoyAI-Echo x LTX-2.5) canvases and the nodes they need.

Every module is imported defensively - one failing module must not take the pack down.
"""
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _merge(modname):
    try:
        mod = __import__(f"{__name__}.{modname}", fromlist=["*"])
    except Exception as e:                                # pragma: no cover
        logging.warning("[JoyLTX] %s not loaded (%s) - its nodes are unavailable; the rest of the pack still works", modname, e)
        return
    NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))


for _m in ("joyltx_loader",      # JoyLTX_LoadModel: .safetensors or .gguf through one node
           "joyltx_controls",    # JoyLTX_TakeControls (VRAM planner) + JoyLTX_AnySwitch
           "joyltx_script",      # JoyLTX_Script: writer shots -> one LTX prompt
           "joyltx_multishot",   # JoyLTX_Multishot: per-shot two-pass + AV-extend joins
           "joyltx_refs",        # JoyLTX_RefsByName: reference photos by character name
           "joyltx_lora",        # JoyLTX_LoraStack (4 slots) + JoyLTX_PromptFile
           "joyltx_source",      # JoyLTX_StorySource: scene idea / idea file / shot file in one panel
           "joyltx_writer_unload"):  # adds the free-VRAM switch to the JoyEcho writer (runtime patch)
    _merge(_m)

try:
    from . import joyltx_patches        # noqa: F401  (LTX x1.5 rational upsampler: partial-load fix)
except Exception as _e:                                   # pragma: no cover
    logging.info("[JoyLTX] upsampler patch skipped (%s)", _e)
