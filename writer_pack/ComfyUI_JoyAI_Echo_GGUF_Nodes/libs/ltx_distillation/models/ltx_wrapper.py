"""
LTX-2 Diffusion Model Wrapper for DMD distillation.

This wrapper adapts LTX-2's audio-video joint generation model for use in
DMD (Distribution Matching Distillation) training.

Model Architecture:
- patch_size = (1, 1, 1): No spatial/temporal grouping
- Patchification: Simple reshape [B, C, F, H, W] → [B, F*H*W, C]
- Each token: 128-dimensional latent vector (one per spatial-temporal position)
- Model input projection: Linear(128, 4096)
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import LTXModel, X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import (
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)


class LTX2DiffusionWrapper(nn.Module):
    """
    Wrapper for LTX-2 model to provide DMD-compatible interface.

    Handles:
    - Input format conversion: [B, F, C, H, W] -> Modality
    - Timestep handling: sigma values for all tokens
    - Position computation for video (3D) and audio (1D)
    - Output format: x0 predictions for both video and audio

    Uses official LTX-2 patchifiers (patch_size=1) to ensure consistency
    with the pretrained model weights.
    """

    # Time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8 (VAE compression)
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4 (mel hop / VAE compression)
    ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS  # ~8.33

    # Video FPS for position computation
    VIDEO_FPS = 24.0

    # VAE scale factors (temporal=8, height=32, width=32)
    DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

    def __init__(
        self,
        model: LTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
    ):
        """
        Args:
            model: X0Model instance (wraps velocity model, returns x0 predictions)
            video_height: Video height in pixels
            video_width: Video width in pixels
            vae_spatial_compression: VAE spatial compression factor
        """
        super().__init__()
        self.model = model
        self.video_height = video_height
        self.video_width = video_width
        self.vae_spatial_compression = vae_spatial_compression

        # Compute latent dimensions
        self.latent_height = video_height // vae_spatial_compression  # 16
        self.latent_width = video_width // vae_spatial_compression    # 24

        # Official patchifiers with patch_size=1 (no spatial grouping)
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

        # Frame sequence length: with patch_size=1, each spatial position is one token
        # For 512x768: H'*W' = 16*24 = 384 tokens per frame
        self.video_frame_seqlen = self.latent_height * self.latent_width  # 384

    def set_module_grad(self, module_grad: Dict[str, bool]) -> None:
        """
        Set gradient requirements for model components.

        Args:
            module_grad: Dict mapping component names to requires_grad flags
        """
        if module_grad.get("model", True):
            self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "velocity_model"):
            self.model.velocity_model.set_gradient_checkpointing(True)
        elif hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(True)

    def _flatten_video_latent(
        self,
        video_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flatten video latent from [B, F, C, H, W] to [B, T, C] using patch_size=1.

        With patch_size=1, this is a simple reshape — no spatial grouping.
        The official VideoLatentPatchifier(patch_size=1) does:
            "b c (f 1) (h 1) (w 1) -> b (f h w) (c 1 1 1)" = "b c f h w -> b (f h w) c"

        Args:
            video_latent: Shape [B, F, C, H, W] where
                - F: number of latent frames
                - C: latent channels (128)
                - H, W: latent spatial dimensions (16, 24)

        Returns:
            Flattened tensor [B, T, C] where:
            - T = F * H * W (e.g., 16 * 16 * 24 = 6144)
            - C = 128 (unchanged, since patch_size=1)
        """
        B, F, C, H, W = video_latent.shape
        assert C == 128, (
            f"Expected video latent C=128 at dim 2, got shape {video_latent.shape}. "
            f"Input should be [B, F, C, H, W] with C=128."
        )

        # Convert from [B, F, C, H, W] to [B, C, F, H, W] (official format)
        video_latent = video_latent.permute(0, 2, 1, 3, 4)

        # Use official patchifier: [B, C, F, H, W] -> [B, F*H*W, C]
        # With patch_size=1 this is equivalent to:
        #   einops.rearrange(x, "b c f h w -> b (f h w) c")
        video_latent = self.video_patchifier.patchify(video_latent)

        return video_latent

    def _unflatten_video_latent(
        self,
        flat_latent: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Unflatten video latent from [B, T, C] back to [B, F, C, H, W].

        Args:
            flat_latent: Shape [B, T, C] where C = 128 (patch_size=1)
            num_frames: Number of latent frames F

        Returns:
            Video latent [B, F, C, H, W]
        """
        B, T, C = flat_latent.shape
        H = self.latent_height
        W = self.latent_width
        F = num_frames

        # Use official unpatchifier: [B, T, C] -> [B, C, F, H, W]
        output_shape = VideoLatentShape(
            batch=B, channels=C, frames=F, height=H, width=W
        )
        video_latent = self.video_patchifier.unpatchify(flat_latent, output_shape)

        # Convert from [B, C, F, H, W] to [B, F, C, H, W] (DMD format)
        video_latent = video_latent.permute(0, 2, 1, 3, 4)

        return video_latent

    def _compute_video_positions(
        self,
        video_latent: torch.Tensor,
        downscale_factor: int = 1,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """
        Compute 3D position indices for video tokens with [start, end) bounds.

        Uses the official VideoLatentPatchifier.get_patch_grid_bounds() and
        get_pixel_coords() to ensure consistency with the pretrained model.

        The RoPE computation expects positions in the format [B, 3, T, 2] where:
        - dim 1 (size 3): temporal, height, width dimensions
        - dim 3 (size 2): [start, end) bounds for each patch

        Returns:
            Position tensor of shape [B, 3, T, 2] with patch bounds in pixel space
        """
        B, F, C, H, W = video_latent.shape
        device = video_latent.device

        # Build VideoLatentShape for the patchifier
        video_shape = VideoLatentShape(
            batch=B, channels=C, frames=F, height=H, width=W
        )

        # Get patch grid bounds in latent coordinates: [B, 3, T, 2]
        # With patch_size=1, each token covers [i, i+1) in each dimension
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=video_shape,
            device=device,
        )
        if start_frame != 0:
            latent_coords = latent_coords.clone()
            latent_coords[:, 0, :, :] += int(start_frame)

        # Convert to pixel coordinates using official helper
        # Applies scale_factors (temporal=8, height=32, width=32)
        # and causal_fix (first frame temporal offset)
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.DEFAULT_SCALE_FACTORS,
            causal_fix=True,
        ).float()

        # Convert temporal dimension from frames to seconds. self.VIDEO_FPS
        # MUST be the actual render fps - the JoyEcho nodes stamp it as an
        # instance attribute before generation. The class default of 24.0 with
        # 25fps renders skewed video rope-time 4% fast against audio's
        # true-seconds rope: a growing mouth-ahead-of-audio drift that crossed
        # visibility at ~9.6s into every shot (the "10s lip-sync cliff",
        # diagnosed 2026-07-23).
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / self.VIDEO_FPS

        if downscale_factor != 1:
            pixel_coords = pixel_coords.clone()
            pixel_coords[:, 1, ...] *= downscale_factor
            pixel_coords[:, 2, ...] *= downscale_factor

        return pixel_coords

    # Audio timing constants (from AudioPatchifier defaults)
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_HOP_LENGTH = 160
    AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
    AUDIO_IS_CAUSAL = True

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts latent indices into real-time seconds while honoring causal
        offsets and the configured hop length.

        Matches AudioPatchifier._get_audio_latent_time_in_sec exactly.
        """
        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.AUDIO_LATENT_DOWNSAMPLE_FACTOR

        if self.AUDIO_IS_CAUSAL:
            # Frame offset for causal alignment.
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.AUDIO_LATENT_DOWNSAMPLE_FACTOR).clip(min=0)

        return audio_mel_frame * self.AUDIO_HOP_LENGTH / self.AUDIO_SAMPLE_RATE

    def _compute_audio_positions(
        self,
        audio_latent: torch.Tensor,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """
        Compute 1D temporal positions for audio tokens with [start, end) bounds.

        The RoPE computation expects positions in the format [B, 1, T, 2] where:
        - dim 1 (size 1): temporal dimension only (audio is 1D)
        - dim 3 (size 2): [start, end) bounds in seconds

        Returns:
            Position tensor of shape [B, 1, T, 2] with temporal bounds in seconds
        """
        B, T, C = audio_latent.shape
        device = audio_latent.device

        # Compute start timings for each audio frame
        start_timings = self._get_audio_latent_time_in_sec(
            int(start_frame), int(start_frame) + T, torch.float32, device
        )
        start_timings = start_timings.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, T]

        # Compute end timings for each audio frame (shifted by 1)
        end_timings = self._get_audio_latent_time_in_sec(
            int(start_frame) + 1, int(start_frame) + T + 1, torch.float32, device
        )
        end_timings = end_timings.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, T]

        # Stack to create [B, 1, T, 2] with [start, end) bounds
        positions = torch.stack([start_timings, end_timings], dim=-1)

        return positions

    def _compute_timesteps_for_tokens(
        self,
        sigma: torch.Tensor,
        num_tokens: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """
        Expand sigma to per-token timesteps.

        In the official pipeline, timesteps = denoise_mask * sigma, producing
        shape [B, T, 1]. Here we replicate sigma to each token belonging to
        the same frame and add a trailing dimension for broadcasting with
        the latent channels.

        Args:
            sigma: Shape [B] or [B, F] - sigma values per frame
            num_tokens: Total number of tokens
            tokens_per_frame: Number of tokens per frame

        Returns:
            Timesteps tensor [B, T, 1] for correct broadcasting with [B, T, C]
        """
        B = sigma.shape[0]

        if sigma.dim() == 1:
            # Single sigma per sample -> expand to all tokens
            return sigma.view(B, 1, 1).expand(B, num_tokens, 1)
        else:
            # Per-frame sigma [B, F] -> expand to per-token [B, T, 1]
            F = sigma.shape[1]
            expanded = sigma.unsqueeze(2).expand(B, F, tokens_per_frame).reshape(B, -1)
            return expanded.unsqueeze(-1)  # [B, T, 1]

    @staticmethod
    def _memory_slot_ranges(total_seq_len: int, num_slots: int) -> list[tuple[int, int]]:
        if total_seq_len <= 0 or num_slots <= 0:
            return []

        ranges: list[tuple[int, int]] = []
        start = 0
        for slot_idx in range(num_slots):
            end = round((slot_idx + 1) * total_seq_len / num_slots)
            if end > start:
                ranges.append((start, end))
            start = end
        return ranges

    @staticmethod
    def _memory_slot_ranges_from_lengths(
        lengths: tuple[int, ...] | None,
        *,
        total_seq_len: int,
        num_slots: int,
    ) -> list[tuple[int, int]]:
        if not lengths or len(lengths) != num_slots:
            return LTX2DiffusionWrapper._memory_slot_ranges(total_seq_len, num_slots)

        ranges: list[tuple[int, int]] = []
        start = 0
        for raw_length in lengths:
            length = max(0, int(raw_length))
            end = min(start + length, total_seq_len)
            if end > start:
                ranges.append((start, end))
            start = end
        if start != total_seq_len:
            return LTX2DiffusionWrapper._memory_slot_ranges(total_seq_len, num_slots)
        return ranges

    @classmethod
    def _build_paired_memory_cross_mask(
        cls,
        *,
        batch_size: int,
        query_memory_seq_len: int,
        query_target_seq_len: int,
        kv_memory_seq_len: int,
        kv_target_seq_len: int,
        num_memory_slots: int,
        device: torch.device,
        query_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
        kv_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
    ) -> torch.Tensor:
        query_total_seq_len = query_memory_seq_len + query_target_seq_len
        kv_total_seq_len = kv_memory_seq_len + kv_target_seq_len
        mask = torch.zeros(
            batch_size,
            query_total_seq_len,
            kv_total_seq_len,
            dtype=torch.bool,
            device=device,
        )

        for batch_idx in range(batch_size):
            query_lengths = (
                query_segment_lengths[batch_idx]
                if query_segment_lengths is not None and batch_idx < len(query_segment_lengths)
                else None
            )
            kv_lengths = (
                kv_segment_lengths[batch_idx]
                if kv_segment_lengths is not None and batch_idx < len(kv_segment_lengths)
                else None
            )
            query_ranges = cls._memory_slot_ranges_from_lengths(
                query_lengths,
                total_seq_len=query_memory_seq_len,
                num_slots=num_memory_slots,
            )
            kv_ranges = cls._memory_slot_ranges_from_lengths(
                kv_lengths,
                total_seq_len=kv_memory_seq_len,
                num_slots=num_memory_slots,
            )
            for (q_start, q_end), (k_start, k_end) in zip(query_ranges, kv_ranges, strict=False):
                mask[batch_idx, q_start:q_end, k_start:k_end] = True

        if query_target_seq_len > 0 and kv_target_seq_len > 0:
            mask[:, query_memory_seq_len:, kv_memory_seq_len:] = True
        return mask

    @staticmethod
    def _build_memory_self_attention_block_mask(
        *,
        batch_size: int,
        memory_seq_len: int,
        target_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if memory_seq_len <= 0:
            return None

        total_seq_len = memory_seq_len + target_seq_len
        attention_mask = torch.ones(
            batch_size,
            total_seq_len,
            total_seq_len,
            dtype=torch.bool,
            device=device,
        )
        attention_mask[:, :, :memory_seq_len] = False
        attention_mask[:, :memory_seq_len, :] = False
        attention_mask[:, :memory_seq_len, :memory_seq_len] = True
        return attention_mask

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        memory_video: Optional[torch.Tensor] = None,
        memory_audio: Optional[torch.Tensor] = None,
        memory_audio_timestep: Optional[torch.Tensor] = None,
        memory_audio_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
        paired_audio_memory: bool = False,
        v2a_grad_scale: float = 1.0,
        memory_position_mode: str = "reference",
        memory_downscale_factor: int = 1,
        skip_a2v_cross_attn: bool = False,
        skip_v2a_cross_attn: bool = False,
        skip_video_self_attn: bool = False,
        skip_audio_self_attn: bool = False,
        use_causal_timestep: bool = False,  # ignored, for API compatibility
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for DMD distillation.

        Args:
            noisy_image_or_video: Noisy video latent [B, F, C, H, W]
            conditional_dict: Dictionary containing:
                - video_context: [B, seq_len, dim]
                - audio_context: [B, seq_len, dim]
                - attention_mask: [B, seq_len]
            timestep: Sigma values [B] or [B, F]
            noisy_audio: Noisy audio latent [B, F_a, C_audio] (optional)
                where C_audio = 128 (= 8 channels * 16 mel_bins, post-patchify)
            audio_timestep: Audio sigma values [B] or [B, F_a] (optional)
            memory_audio: Optional clean memory-audio prefix [B, F_mem_a, C_audio]
            memory_audio_timestep: Optional memory-audio sigma values [B] or [B, F_mem_a]

        Returns:
            Tuple of (video_x0_pred, audio_x0_pred)
            - video_x0_pred: [B, F, C, H, W]
            - audio_x0_pred: [B, F_a, C_audio] or None
        """
        B = noisy_image_or_video.shape[0]
        num_video_frames = noisy_image_or_video.shape[1]
        device = noisy_image_or_video.device
        memory_position_mode = str(memory_position_mode).lower()
        if memory_position_mode == "reference":
            memory_position_mode = "legacy"
        if memory_position_mode not in {"legacy", "prefix_continuous"}:
            raise ValueError(
                "memory_position_mode must be one of {'reference', 'legacy', 'prefix_continuous'}, "
                f"got {memory_position_mode}"
            )
        if memory_video is not None and int(memory_video.shape[1]) == 0:
            memory_video = None

        # Flatten target video latent: [B, F, C, H, W] -> [B, T, C]
        # With patch_size=1: T = F*H*W, C = 128
        target_video_flat = self._flatten_video_latent(noisy_image_or_video)
        num_target_video_tokens = target_video_flat.shape[1]

        # Compute target video positions / timesteps
        target_video_position_start = (
            int(memory_video.shape[1])
            if memory_position_mode == "prefix_continuous" and memory_video is not None
            else 0
        )
        target_video_positions = self._compute_video_positions(
            noisy_image_or_video,
            start_frame=target_video_position_start,
        )
        target_video_timesteps = self._compute_timesteps_for_tokens(
            timestep, num_target_video_tokens, self.video_frame_seqlen
        )

        memory_seq_len = 0
        if memory_video is not None:
            memory_video_flat = self._flatten_video_latent(memory_video)
            memory_video_positions = self._compute_video_positions(
                memory_video, downscale_factor=memory_downscale_factor
            )
            memory_video_timesteps = torch.zeros(
                B,
                memory_video_flat.shape[1],
                1,
                device=device,
                dtype=target_video_timesteps.dtype,
            )
            video_flat = torch.cat([memory_video_flat, target_video_flat], dim=1)
            video_positions = torch.cat([memory_video_positions, target_video_positions], dim=2)
            video_timesteps = torch.cat([memory_video_timesteps, target_video_timesteps], dim=1)
            memory_seq_len = memory_video_flat.shape[1]
        else:
            video_flat = target_video_flat
            video_positions = target_video_positions
            video_timesteps = target_video_timesteps

        # Build video modality
        video_sigma = timestep if timestep.dim() == 1 else timestep[:, 0]
        video_modality = Modality(
            latent=video_flat,
            sigma=video_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=conditional_dict["video_context"],
            context_mask=conditional_dict.get("attention_mask"),
            enabled=True,
        )

        # Build audio modality if provided
        audio_modality = None
        memory_audio_seq_len = 0
        if noisy_audio is None and (memory_audio is not None or memory_audio_timestep is not None):
            raise ValueError("memory_audio requires noisy_audio")
        if noisy_audio is not None:
            target_audio = noisy_audio
            target_audio_frames = target_audio.shape[1]

            # Use provided audio timestep or derive from video timestep
            if audio_timestep is None:
                # In bidirectional mode, audio uses same sigma as video.
                # video timestep could be [B] or [B, F_v]. For audio we need [B]
                # or [B, F_a]. If timestep is [B, F_v] (per-frame video), take the
                # first frame's sigma since bidirectional uses uniform sigma anyway.
                if timestep.dim() == 1:
                    audio_timestep = timestep  # [B]
                else:
                    # All video frames have same sigma in bidirectional mode,
                    # take the first frame's value and broadcast to audio frames
                    audio_timestep = timestep[:, 0]  # [B]

            if audio_timestep.dim() == 1:
                target_audio_timestep = audio_timestep[:, None].expand(B, target_audio_frames)
            elif audio_timestep.shape == (B, target_audio_frames):
                target_audio_timestep = audio_timestep
            else:
                raise ValueError(
                    "audio_timestep must have shape [B] or [B, F_a], "
                    f"got {tuple(audio_timestep.shape)} vs {(B, target_audio_frames)}"
                )

            if memory_audio_timestep is not None and memory_audio is None:
                raise ValueError("memory_audio_timestep requires memory_audio")

            if memory_audio is not None:
                memory_audio = memory_audio.to(device=device, dtype=target_audio.dtype)
                memory_audio_seq_len = memory_audio.shape[1]
                if memory_audio_timestep is None:
                    prefix_audio_timestep = torch.zeros(
                        B,
                        memory_audio_seq_len,
                        device=device,
                        dtype=target_audio_timestep.dtype,
                    )
                elif memory_audio_timestep.dim() == 1:
                    prefix_audio_timestep = memory_audio_timestep[:, None].expand(B, memory_audio_seq_len)
                elif memory_audio_timestep.shape == (B, memory_audio_seq_len):
                    prefix_audio_timestep = memory_audio_timestep
                else:
                    raise ValueError(
                        "memory_audio_timestep must have shape [B] or [B, F_mem_a], "
                        f"got {tuple(memory_audio_timestep.shape)} vs {(B, memory_audio_seq_len)}"
                    )

                noisy_audio = torch.cat([memory_audio, target_audio], dim=1)
                combined_audio_timestep = torch.cat([prefix_audio_timestep, target_audio_timestep], dim=1)
            else:
                noisy_audio = target_audio
                combined_audio_timestep = target_audio_timestep

            num_audio_tokens = noisy_audio.shape[1]
            audio_timesteps = self._compute_timesteps_for_tokens(combined_audio_timestep, num_audio_tokens, 1)
            if memory_audio_seq_len > 0:
                memory_audio_positions = self._compute_audio_positions(memory_audio)
                target_audio_position_start = memory_audio_seq_len if memory_position_mode == "prefix_continuous" else 0
                target_audio_positions = self._compute_audio_positions(
                    target_audio,
                    start_frame=target_audio_position_start,
                )
                audio_positions = torch.cat([memory_audio_positions, target_audio_positions], dim=2)
            else:
                audio_positions = self._compute_audio_positions(noisy_audio)
            audio_sigma = target_audio_timestep[:, 0]
            audio_modality = Modality(
                latent=noisy_audio,
                sigma=audio_sigma,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=conditional_dict.get("audio_context", conditional_dict["video_context"]),
                context_mask=conditional_dict.get("attention_mask"),
                enabled=True,
                v2a_grad_scale=float(v2a_grad_scale),
            )

        if bool(paired_audio_memory) and memory_seq_len > 0 and audio_modality is not None and memory_audio_seq_len > 0:
            num_memory_slots = int(memory_video.shape[1]) if memory_video is not None else 0
            if num_memory_slots > 0:
                target_audio_seq_len = int(audio_modality.latent.shape[1] - memory_audio_seq_len)
                a2v_pairwise_mask = self._build_paired_memory_cross_mask(
                    batch_size=B,
                    query_memory_seq_len=memory_seq_len,
                    query_target_seq_len=num_target_video_tokens,
                    kv_memory_seq_len=memory_audio_seq_len,
                    kv_target_seq_len=target_audio_seq_len,
                    num_memory_slots=num_memory_slots,
                    device=device,
                    kv_segment_lengths=memory_audio_segment_lengths,
                )
                v2a_pairwise_mask = self._build_paired_memory_cross_mask(
                    batch_size=B,
                    query_memory_seq_len=memory_audio_seq_len,
                    query_target_seq_len=target_audio_seq_len,
                    kv_memory_seq_len=memory_seq_len,
                    kv_target_seq_len=num_target_video_tokens,
                    num_memory_slots=num_memory_slots,
                    device=device,
                    query_segment_lengths=memory_audio_segment_lengths,
                )
                video_cross_query_mask = torch.ones(
                    B,
                    video_modality.latent.shape[1],
                    device=device,
                    dtype=torch.bool,
                )
                audio_cross_query_mask = torch.ones(
                    B,
                    audio_modality.latent.shape[1],
                    device=device,
                    dtype=torch.bool,
                )
                audio_attention_mask = self._build_memory_self_attention_block_mask(
                    batch_size=B,
                    memory_seq_len=memory_audio_seq_len,
                    target_seq_len=target_audio_seq_len,
                    device=device,
                )
                video_modality = replace(
                    video_modality,
                    cross_kv_mask=v2a_pairwise_mask,
                    cross_query_mask=video_cross_query_mask,
                    late_cross_kv_mask=v2a_pairwise_mask,
                    late_cross_query_mask=video_cross_query_mask,
                )
                audio_modality = replace(
                    audio_modality,
                    attention_mask=audio_attention_mask,
                    cross_kv_mask=a2v_pairwise_mask,
                    cross_query_mask=audio_cross_query_mask,
                    late_cross_kv_mask=a2v_pairwise_mask,
                    late_cross_query_mask=audio_cross_query_mask,
                )

        # Forward through model. The optional perturbation flags let inference
        # freeze one direction of cross-modal interaction without modifying the
        # shared core transformer implementation.
        perturbation_items: list[Perturbation] = []
        if skip_a2v_cross_attn:
            perturbation_items.append(
                Perturbation(
                    type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                    blocks=None,
                )
            )
        if skip_v2a_cross_attn:
            perturbation_items.append(
                Perturbation(
                    type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                    blocks=None,
                )
            )
        if skip_video_self_attn:
            perturbation_items.append(
                Perturbation(
                    type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    blocks=None,
                )
            )
        if skip_audio_self_attn:
            perturbation_items.append(
                Perturbation(
                    type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    blocks=None,
                )
            )

        if perturbation_items:
            perturbation_config = PerturbationConfig(perturbations=perturbation_items)
            perturbations = BatchedPerturbationConfig(
                [perturbation_config for _ in range(B)]
            )
        else:
            perturbations = BatchedPerturbationConfig.empty(batch_size=B)

        # The model returns x0 predictions (X0Model wraps velocity model)
        video_x0, audio_x0 = self.model(
            video=video_modality,
            audio=audio_modality,
            perturbations=perturbations,
        )

        # Unflatten video output: [B, T, C] -> [B, F, C, H, W]
        if video_x0 is not None:
            if memory_seq_len > 0:
                video_x0 = video_x0[:, memory_seq_len:, :]
            video_x0 = self._unflatten_video_latent(video_x0, num_video_frames)
        if audio_x0 is not None and memory_audio_seq_len > 0:
            audio_x0 = audio_x0[:, memory_audio_seq_len:, :]

        return video_x0, audio_x0

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state dict, handling potential key mismatches."""
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k] = v
            else:
                new_state_dict[f"model.{k}"] = v

        super().load_state_dict(new_state_dict, strict=strict)


def create_ltx2_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_height: int = 512,
    video_width: int = 768,
    loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
    registry: Registry | None = None,
    quantization=None,
) -> LTX2DiffusionWrapper:
    """
    Factory function to create LTX2DiffusionWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype
        video_height: Video height
        video_width: Video width

    Returns:
        Configured LTX2DiffusionWrapper
    """
    from ltx_pipelines.utils.model_ledger import ModelLedger

    # IMPORTANT: Load to CPU first, then move to target device
    # safetensors doesn't support device indices like "cuda:4"
    # It only accepts "cuda" or "cpu"
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),  # Load to CPU first
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        loras=loras,
        registry=registry,
        quantization=quantization,
    )

    # Get X0Model (wraps velocity model)
    x0_model = ledger.transformer()

    # Move to target device. A dtype cast here would silently upcast
    # quantized fp8 weights back to bf16, so only move the device when a
    # quantization policy is active.
    if quantization is None:
        x0_model = x0_model.to(device=device, dtype=dtype)
    else:
        x0_model = x0_model.to(device=device)
        # Some checkpoints (e.g. plain distilled-1.1 and files copied from
        # it) store modulation tables in float32. Skipping the global cast
        # protects fp8 weights, but a float32 parameter anywhere poisons the
        # compute dtype downstream (f32 activations hitting bf16 linears ->
        # "mat1 and mat2 must have the same dtype"). Cast exactly the
        # non-quantized float32 leftovers; fp8 storage is untouched.
        n_f32 = 0
        for p in x0_model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(dtype)
                n_f32 += 1
        for b in x0_model.buffers():
            if b.dtype == torch.float32:
                b.data = b.data.to(dtype)
                n_f32 += 1
        if n_f32:
            print(f"[ltx_wrapper] quantized load: cast {n_f32} float32 "
                  f"tensors -> {dtype} (global cast is skipped under a "
                  f"quantization policy).", flush=True)

    wrapper = LTX2DiffusionWrapper(
        model=x0_model,
        video_height=video_height,
        video_width=video_width,
    )

    return wrapper
