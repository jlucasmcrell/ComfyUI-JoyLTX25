"""
Bidirectional pipelines for memory-conditioned DMD.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BidirectionalMemoryVideoTrajectoryPipeline:
    """
    Few-step backward simulation for video-only memory-conditioned DMD.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        memory_downscale_factor: int = 1,
        audio_latent_clamp: float = 0.0,
    ) -> None:
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.memory_downscale_factor = int(memory_downscale_factor)
        self.audio_latent_clamp = float(audio_latent_clamp)

    @torch.no_grad()
    def inference_with_trajectory(
        self,
        video_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
        memory_video: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = video_noise.shape[0]
        num_frames = video_noise.shape[1]
        device = video_noise.device
        dtype = video_noise.dtype
        memory_video = memory_video.to(device=device, dtype=dtype)

        trajectory = [video_noise]
        noisy_video = video_noise

        for idx, sigma in enumerate(self.denoising_sigmas[:-1]):
            video_sigma = sigma * torch.ones([batch_size, num_frames], device=device, dtype=dtype)
            pred_video, _ = self.generator(
                noisy_image_or_video=noisy_video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=None,
                audio_timestep=None,
                memory_video=memory_video,
                memory_downscale_factor=self.memory_downscale_factor,
            )
            pred_video = pred_video.to(dtype=dtype)

            next_sigma = self.denoising_sigmas[idx + 1]
            if next_sigma > 0:
                fresh_noise = torch.randn_like(video_noise)
                next_video_sigma = next_sigma * torch.ones([batch_size, num_frames], device=device, dtype=dtype)
                noisy_video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (batch_size, num_frames)).to(dtype=dtype)
            else:
                noisy_video = pred_video

            trajectory.append(noisy_video)

        return torch.stack(trajectory, dim=1)


class BidirectionalMemoryVideoInferencePipeline:
    """
    Few-step benchmark/inference pipeline for video-only memory-conditioned DMD.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        memory_downscale_factor: int = 1,
        trace_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.memory_downscale_factor = int(memory_downscale_factor)
        self.trace_fn = trace_fn

    def _emit_trace(
        self,
        event: str,
        tensor: torch.Tensor,
        *,
        sigma_idx: Optional[int] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> None:
        if self.trace_fn is None:
            return
        values = tensor.detach().float()
        if values.numel() == 0:
            stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "absmax": 0.0,
                "nonzero_frac": 0.0,
            }
        else:
            stats = {
                "mean": values.mean().item(),
                "std": values.std(unbiased=False).item() if values.numel() > 1 else 0.0,
                "min": values.min().item(),
                "max": values.max().item(),
                "absmax": values.abs().max().item(),
                "nonzero_frac": values.ne(0).float().mean().item(),
            }
        payload: Dict[str, Any] = {
            "phase": "bootstrap",
            "event": event,
            "shape": list(tensor.shape),
            **stats,
        }
        if sigma_idx is not None:
            payload["sigma_idx"] = int(sigma_idx)
        if sigma is not None:
            payload["sigma"] = float(sigma.detach().float().item())
        self.trace_fn(payload)

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        conditional_dict: Dict[str, Any],
        memory_video: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = video_shape[0]
        num_frames = video_shape[1]

        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.generator.parameters()).device
        dtype = next((p.dtype for p in self.generator.parameters()
                     if p.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)),
                    torch.bfloat16)  # skip fp8 params: randn/noise math needs a real float dtype

        video = torch.randn(video_shape, device=device, dtype=dtype)
        memory_video = memory_video.to(device=device, dtype=dtype)
        self._emit_trace("initial_noise", video)

        for idx, sigma in enumerate(self.denoising_sigmas[:-1]):
            video_sigma = sigma * torch.ones([batch_size, num_frames], device=device, dtype=dtype)

            pred_video, _ = self.generator(
                noisy_image_or_video=video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=None,
                audio_timestep=None,
                memory_video=memory_video,
                memory_downscale_factor=self.memory_downscale_factor,
            )
            pred_video = pred_video.to(dtype=dtype)
            self._emit_trace("pred_x0", pred_video, sigma_idx=idx, sigma=sigma)

            next_sigma = self.denoising_sigmas[idx + 1]
            if next_sigma > 0:
                fresh_noise = torch.randn_like(video)
                next_video_sigma = next_sigma * torch.ones([batch_size, num_frames], device=device, dtype=dtype)
                video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (batch_size, num_frames)).to(dtype=dtype)
            else:
                video = pred_video
            self._emit_trace("updated_video", video, sigma_idx=idx + 1, sigma=next_sigma)

        return video


class BidirectionalMemoryAVTrajectoryPipeline:
    """
    Few-step backward simulation for video-memory-conditioned AV DMD.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        memory_downscale_factor: int = 1,
        audio_latent_clamp: float = 0.0,
    ) -> None:
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.memory_downscale_factor = int(memory_downscale_factor)
        self.audio_latent_clamp = float(audio_latent_clamp)

    @torch.no_grad()
    def inference_with_trajectory(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
        memory_video: torch.Tensor,
        memory_audio: Optional[torch.Tensor] = None,
        memory_audio_timestep: Optional[torch.Tensor] = None,
        memory_audio_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
        paired_audio_memory: bool = False,
        v2a_grad_scale: float = 1.0,
        memory_position_mode: str = "reference",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = video_noise.shape[0]
        num_video_frames = video_noise.shape[1]
        num_audio_frames = audio_noise.shape[1]
        device = video_noise.device
        dtype = video_noise.dtype
        memory_video = memory_video.to(device=device, dtype=dtype)
        if memory_audio is not None:
            memory_audio = memory_audio.to(device=device, dtype=dtype)
        if memory_audio_timestep is not None:
            memory_audio_timestep = memory_audio_timestep.to(device=device, dtype=dtype)

        video_trajectory = [video_noise]
        audio_trajectory = [audio_noise]
        noisy_video = video_noise
        noisy_audio = audio_noise
        memory_audio_kwargs = (
            {
                "memory_audio": memory_audio,
                "memory_audio_timestep": memory_audio_timestep,
            }
            if memory_audio is not None or memory_audio_timestep is not None
            else {}
        )
        paired_memory_kwargs = (
            {
                "memory_audio_segment_lengths": memory_audio_segment_lengths,
                "paired_audio_memory": True,
                "v2a_grad_scale": v2a_grad_scale,
                "memory_position_mode": memory_position_mode,
            }
            if paired_audio_memory
            else {}
        )

        for idx, sigma in enumerate(self.denoising_sigmas[:-1]):
            video_sigma = sigma * torch.ones([batch_size, num_video_frames], device=device, dtype=dtype)
            audio_sigma = sigma * torch.ones([batch_size, num_audio_frames], device=device, dtype=dtype)

            pred_video, pred_audio = self.generator(
                noisy_image_or_video=noisy_video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=noisy_audio,
                audio_timestep=audio_sigma,
                memory_video=memory_video,
                memory_downscale_factor=self.memory_downscale_factor,
                **memory_audio_kwargs,
                **paired_memory_kwargs,
            )
            pred_video = pred_video.to(dtype=dtype)
            pred_audio = pred_audio.to(dtype=dtype)
            if self.audio_latent_clamp > 0:
                pred_audio = pred_audio.clamp(-self.audio_latent_clamp, self.audio_latent_clamp)

            next_sigma = self.denoising_sigmas[idx + 1]
            if next_sigma > 0:
                fresh_noise_video = torch.randn_like(video_noise)
                fresh_noise_audio = torch.randn_like(audio_noise)
                next_video_sigma = next_sigma * torch.ones([batch_size, num_video_frames], device=device, dtype=dtype)
                next_audio_sigma = next_sigma * torch.ones([batch_size, num_audio_frames], device=device, dtype=dtype)
                noisy_video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise_video.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (batch_size, num_video_frames)).to(dtype=dtype)
                noisy_audio = self.add_noise_fn(
                    pred_audio,
                    fresh_noise_audio,
                    next_audio_sigma,
                ).to(dtype=dtype)
            else:
                noisy_video = pred_video
                noisy_audio = pred_audio

            video_trajectory.append(noisy_video)
            audio_trajectory.append(noisy_audio)

        return torch.stack(video_trajectory, dim=1), torch.stack(audio_trajectory, dim=1)


class BidirectionalMemoryAVInferencePipeline:
    """
    Few-step benchmark/inference pipeline for video-memory-conditioned AV generation.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        memory_downscale_factor: int = 1,
    ) -> None:
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.memory_downscale_factor = int(memory_downscale_factor)

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        conditional_dict: Dict[str, Any],
        memory_video: torch.Tensor,
        memory_audio: Optional[torch.Tensor] = None,
        memory_audio_timestep: Optional[torch.Tensor] = None,
        memory_audio_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
        paired_audio_memory: bool = False,
        v2a_grad_scale: float = 1.0,
        memory_position_mode: str = "reference",
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = video_shape[0]
        num_video_frames = video_shape[1]
        num_audio_frames = audio_shape[1]

        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.generator.parameters()).device
        dtype = next((p.dtype for p in self.generator.parameters()
                     if p.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)),
                    torch.bfloat16)  # skip fp8 params: randn/noise math needs a real float dtype

        video = torch.randn(video_shape, device=device, dtype=dtype)
        audio = torch.randn(audio_shape, device=device, dtype=dtype)
        memory_video = memory_video.to(device=device, dtype=dtype)
        if memory_audio is not None:
            memory_audio = memory_audio.to(device=device, dtype=dtype)
        if memory_audio_timestep is not None:
            memory_audio_timestep = memory_audio_timestep.to(device=device, dtype=dtype)
        memory_audio_kwargs = (
            {
                "memory_audio": memory_audio,
                "memory_audio_timestep": memory_audio_timestep,
            }
            if memory_audio is not None or memory_audio_timestep is not None
            else {}
        )
        paired_memory_kwargs = (
            {
                "memory_audio_segment_lengths": memory_audio_segment_lengths,
                "paired_audio_memory": True,
                "v2a_grad_scale": v2a_grad_scale,
                "memory_position_mode": memory_position_mode,
            }
            if paired_audio_memory
            else {}
        )

        for idx, sigma in enumerate(self.denoising_sigmas[:-1]):
            video_sigma = sigma * torch.ones([batch_size, num_video_frames], device=device, dtype=dtype)
            audio_sigma = sigma * torch.ones([batch_size, num_audio_frames], device=device, dtype=dtype)

            pred_video, pred_audio = self.generator(
                noisy_image_or_video=video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=audio,
                audio_timestep=audio_sigma,
                memory_video=memory_video,
                memory_downscale_factor=self.memory_downscale_factor,
                **memory_audio_kwargs,
                **paired_memory_kwargs,
            )
            pred_video = pred_video.to(dtype=dtype)
            pred_audio = pred_audio.to(dtype=dtype)

            next_sigma = self.denoising_sigmas[idx + 1]
            if next_sigma > 0:
                fresh_noise_video = torch.randn_like(video)
                fresh_noise_audio = torch.randn_like(audio)
                next_video_sigma = next_sigma * torch.ones([batch_size, num_video_frames], device=device, dtype=dtype)
                next_audio_sigma = next_sigma * torch.ones([batch_size, num_audio_frames], device=device, dtype=dtype)
                video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise_video.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (batch_size, num_video_frames)).to(dtype=dtype)
                audio = self.add_noise_fn(pred_audio, fresh_noise_audio, next_audio_sigma).to(dtype=dtype)
            else:
                video = pred_video
                audio = pred_audio

        return video, audio
