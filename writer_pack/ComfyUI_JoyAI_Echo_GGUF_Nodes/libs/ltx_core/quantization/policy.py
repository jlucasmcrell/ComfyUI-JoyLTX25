from dataclasses import dataclass

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.quantization.fp8_cast import TRANSFORMER_LINEAR_DOWNCAST_MAP, UPCAST_DURING_INFERENCE
from ltx_core.quantization.fp8_scaled_mm import FP8_PREPARE_MODULE_OPS, FP8_TRANSPOSE_SD_OPS


@dataclass(frozen=True)
class QuantizationPolicy:
    """Configuration for model quantization during loading.
    Attributes:
        sd_ops: State dict operations for weight transformation.
        module_ops: Post-load module transformations.
    """

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()

    @classmethod
    def fp8_cast(cls) -> "QuantizationPolicy":
        """Create policy using FP8 casting with upcasting during inference."""
        return cls(
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(UPCAST_DURING_INFERENCE,),
        )

    @classmethod
    def fp8_scaled_mm(cls) -> "QuantizationPolicy":
        """Create policy using FP8 scaled matrix multiplication."""
        try:
            import tensorrt_llm  # noqa: F401, PLC0415
        except ImportError as e:
            raise ImportError("tensorrt_llm is not installed, skipping FP8 scaled MM quantization") from e

        return cls(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    @classmethod
    def fp8_scaled_mm_torch(cls) -> "QuantizationPolicy":
        """FP8 storage + torch._scaled_mm compute (no tensorrt_llm needed).

        Same load-time downcast as fp8_cast(), but matmuls execute natively in
        fp8 on sm_89+ GPUs instead of upcasting per layer - removing the
        upcast tax and (weights resident) the offload streaming cost.
        """
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("this torch build has no torch._scaled_mm; use fp8_cast instead")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap < (8, 9):
                raise RuntimeError(
                    f"fp8_scaled_mm needs native fp8 tensor cores (sm_89+, i.e. RTX 40/50-series); "
                    f"this GPU is sm_{cap[0]}{cap[1]} (e.g. RTX 30-series). Turn fp8_scaled_mm OFF "
                    f"on this machine - use fp8_transformer for fp8 storage, or plain bf16.")
        from ltx_core.quantization.fp8_torch_mm import TORCH_SCALED_MM_DURING_INFERENCE  # noqa: PLC0415

        return cls(
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(TORCH_SCALED_MM_DURING_INFERENCE,),
        )
