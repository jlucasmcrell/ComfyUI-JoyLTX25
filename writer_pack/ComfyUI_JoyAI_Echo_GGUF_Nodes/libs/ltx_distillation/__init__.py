"""
LTX-2 Distillation Package.

Keep package import side effects light so submodules can be imported in
isolation without eagerly loading the full dependency chain.
"""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "LTX2DMD",
    "get_denoising_loss",
    # ODE Init
    "LTX2ODEPairGenerator",
    "ODEGenerationConfig",
    "LTX2ODERegression",
    "ODERegressionConfig",
    "ODERegressionLMDBDataset",
    "ODERegressionDataset",
]

_LAZY_IMPORTS = {
    "LTX2DMD": ("ltx_distillation.dmd", "LTX2DMD"),
    "get_denoising_loss": ("ltx_distillation.loss", "get_denoising_loss"),
    "LTX2ODEPairGenerator": ("ltx_distillation.ode", "LTX2ODEPairGenerator"),
    "ODEGenerationConfig": ("ltx_distillation.ode", "ODEGenerationConfig"),
    "LTX2ODERegression": ("ltx_distillation.ode", "LTX2ODERegression"),
    "ODERegressionConfig": ("ltx_distillation.ode", "ODERegressionConfig"),
    "ODERegressionLMDBDataset": ("ltx_distillation.ode", "ODERegressionLMDBDataset"),
    "ODERegressionDataset": ("ltx_distillation.ode", "ODERegressionDataset"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
