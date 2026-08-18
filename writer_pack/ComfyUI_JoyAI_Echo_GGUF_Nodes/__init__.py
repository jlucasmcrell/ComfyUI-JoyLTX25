"""ComfyUI nodes for JoyAI-Echo: minute-level multi-shot audio-video generation.

Registers:
  - the 7 upstream JoyEcho nodes (nodes.py)
  - the discrete Rebels loader nodes (rebels_loaders.py)
  - the Rebels staged single-node pipeline for 16GB RAM (rebels_staged.py)

The libs/ folder MUST be on sys.path before anything imports ltx_core /
ltx_distillation, so that setup happens first.
"""

import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
_LIBS = str(_NODE_ROOT / "libs")

if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# ------------------------------------------------------------ base-pack check
# This pack is a PATCH (overlay) on RealRebelAI's ComfyUI_JoyAI_Echo_GGUF_Nodes.
# It ships only the files it modifies; base support modules like
# text_encoder_wrapper.py and vae_wrapper.py must already be present. If a user
# REPLACES the folder instead of merging the patch INTO it, those base files are
# deleted - the nodes still register (their imports are lazy) but the first
# render dies with a cryptic "No module named ..." mid-run. Catch it at startup
# with an actionable message instead.
_REQUIRED_BASE = [
    "libs/ltx_distillation/models/text_encoder_wrapper.py",
    "libs/ltx_distillation/models/vae_wrapper.py",
    "libs/ltx_core/model/transformer/transformer.py",
]
_missing = [f for f in _REQUIRED_BASE if not (_NODE_ROOT / f).is_file()]
if _missing:
    _msg = (
        "\n" + "=" * 74 +
        "\n[JoyEcho] INSTALL PROBLEM: base-pack files are missing:\n  " +
        "\n  ".join(_missing) +
        "\n\nThis pack is a PATCH on top of RealRebelAI's "
        "ComfyUI_JoyAI_Echo_GGUF_Nodes.\nYou likely REPLACED the folder instead "
        "of MERGING the patch into it.\nFix: reinstall the base pack (git clone "
        "RealRebelAI's), then copy the\npatch files ON TOP, choosing "
        "MERGE / 'replace files in destination' - never\ndelete-and-replace the "
        "whole folder. Renders will fail until this is fixed.\n" + "=" * 74)
    print(_msg, flush=True)

# ---------------------------------------------------------------- base maps
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ---------------------------------------------------------------- upstream JoyEcho nodes
from .nodes import (
    JoyEcho_ModelLoader,
    JoyEcho_LoraStacker,
    JoyEcho_TextEncode,
    JoyEcho_Generate,
    JoyEcho_SingleShotGenerate,
    JoyEcho_PromptFormat,
    JoyEcho_LLMEnhance,
    JoyEcho_PromptAtIndex,
)

NODE_CLASS_MAPPINGS.update({
    "JoyEcho_ModelLoader": JoyEcho_ModelLoader,
    "JoyEcho_LoraStacker": JoyEcho_LoraStacker,
    "JoyEcho_TextEncode": JoyEcho_TextEncode,
    "JoyEcho_Generate": JoyEcho_Generate,
    "JoyEcho_SingleShotGenerate": JoyEcho_SingleShotGenerate,
    "JoyEcho_PromptFormat": JoyEcho_PromptFormat,
    "JoyEcho_LLMEnhance": JoyEcho_LLMEnhance,
    "JoyEcho_PromptAtIndex": JoyEcho_PromptAtIndex,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "JoyEcho_ModelLoader": "JoyEcho Model Loader",
    "JoyEcho_LoraStacker": "JoyEcho LoRA Stack",
    "JoyEcho_TextEncode": "JoyEcho Text Encode",
    "JoyEcho_Generate": "JoyEcho Generate (Multi-Shot)",
    "JoyEcho_SingleShotGenerate": "JoyEcho Single Shot Generate",
    "JoyEcho_PromptFormat": "JoyEcho Prompt Format (Helper)",
    "JoyEcho_LLMEnhance": "JoyEcho LLM Enhance",
    "JoyEcho_PromptAtIndex": "JoyEcho Prompt At Index",
})

# ---------------------------------------------------------------- Rebels discrete loaders
try:
    from .rebels_loaders import (
        NODE_CLASS_MAPPINGS as _RL_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _RL_DM,
    )
    NODE_CLASS_MAPPINGS.update(_RL_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RL_DM)
except Exception as e:
    print(f"[Rebels JE] rebels_loaders failed to load: {e!r}", flush=True)

# ---------------------------------------------------------------- Rebels staged pipeline (16GB)
try:
    # Auto-inject a verified surface plate as reference_image, chosen from the
    # RE-CAPTIONED plate library by the prompt's ground material. Optional and
    # fail-open: if the module or the library is missing, nothing else breaks.
    from .joyecho_plate_picker import (
        NODE_CLASS_MAPPINGS as _PP_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _PP_DM,
    )
    NODE_CLASS_MAPPINGS.update(_PP_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_PP_DM)
except Exception as _e_pp:
    print(f"[JoyEcho] PlatePicker unavailable ({type(_e_pp).__name__}: {_e_pp})",
          flush=True)

try:
    from .rebels_staged import (
        NODE_CLASS_MAPPINGS as _ST_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _ST_DM,
    )
    NODE_CLASS_MAPPINGS.update(_ST_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ST_DM)
except Exception as e:
    print(f"[Rebels JE] rebels_staged failed to load: {e!r}", flush=True)

# ---------------------------------------------------------------- Script picker (JSON dropdown)
# joyecho_script_picker moved to ComfyUI-H3-Multishot as rift_script_picker.py (v2.1); it registers the old
# JoyEcho_ key as a deprecated alias so saved graphs still open.

# ---------------------------------------------------------------- Prompt source (unified txt+json dropdown)
# joyecho_prompt_source moved to ComfyUI-H3-Multishot as rift_prompt_source.py (v2.1); it registers the old
# JoyEcho_ key as a deprecated alias so saved graphs still open.

# ---------------------------------------------------------------- Reference batch (None-tolerant)
try:
    from .joyecho_ref_batch import (
        NODE_CLASS_MAPPINGS as _RB_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _RB_DM,
    )
    NODE_CLASS_MAPPINGS.update(_RB_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RB_DM)
except Exception as e:
    print(f"[Rebels JE] joyecho_ref_batch failed to load: {e!r}", flush=True)

# ---------------------------------------------------------------- Reference picker (auto by character)
try:
    from .joyecho_ref_picker import (
        NODE_CLASS_MAPPINGS as _RP_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _RP_DM,
    )
    NODE_CLASS_MAPPINGS.update(_RP_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RP_DM)
except Exception as e:
    print(f"[Rebels JE] joyecho_ref_picker failed to load: {e!r}", flush=True)

# ---------------------------------------------------------------- Auto-finish (RTX upscale + master concat after render)
try:
    from .joyecho_autofinish import (
        NODE_CLASS_MAPPINGS as _AF_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _AF_DM,
    )
    NODE_CLASS_MAPPINGS.update(_AF_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_AF_DM)
except Exception as e:
    print(f"[Rebels JE] joyecho_autofinish failed to load: {e!r}", flush=True)

try:
    from .joyecho_cartridge import (
        NODE_CLASS_MAPPINGS as _CG_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _CG_DM,
    )
    NODE_CLASS_MAPPINGS.update(_CG_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_CG_DM)
except Exception as _e:
    print(f"[JoyAI-Echo] cartridge loader unavailable: {_e}")

try:
    from .riftcast_generator import (
        NODE_CLASS_MAPPINGS as _RG_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _RG_DM,
    )
    NODE_CLASS_MAPPINGS.update(_RG_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RG_DM)
except Exception as _e:
    print(f"[JoyAI-Echo] RiftCast generator unavailable: {_e}")

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
