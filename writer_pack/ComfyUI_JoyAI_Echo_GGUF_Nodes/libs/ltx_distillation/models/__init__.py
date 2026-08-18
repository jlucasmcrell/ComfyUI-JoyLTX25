"""
Model wrappers for LTX-2 distillation.

Use lazy imports so submodules can be imported without eagerly loading the VAE
stack and its optional runtime dependencies.
"""

from importlib import import_module

__all__ = [
    "LTX2DiffusionWrapper",
    "GemmaTextEncoderWrapper",
    "VideoVAEWrapper",
    "AudioVAEWrapper",
]

_LAZY_IMPORTS = {
    "LTX2DiffusionWrapper": ("ltx_distillation.models.ltx_wrapper", "LTX2DiffusionWrapper"),
    "GemmaTextEncoderWrapper": ("ltx_distillation.models.text_encoder_wrapper", "GemmaTextEncoderWrapper"),
    "VideoVAEWrapper": ("ltx_distillation.models.vae_wrapper", "VideoVAEWrapper"),
    "AudioVAEWrapper": ("ltx_distillation.models.vae_wrapper", "AudioVAEWrapper"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
