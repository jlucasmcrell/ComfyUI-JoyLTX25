"""Utilities for selecting paired audio/video memory windows."""

from __future__ import annotations

import math
import random
from typing import Literal

import torch
from torch import Tensor


def latent_window_size_to_pixel_window_size(
    latent_window_size: int,
    *,
    downsample_factor: int,
    is_causal: bool = True,
) -> int:
    if latent_window_size <= 0:
        raise ValueError(f"latent_window_size must be positive, got {latent_window_size}")
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be positive, got {downsample_factor}")

    pixel_window_size = int(latent_window_size) * int(downsample_factor)
    if is_causal:
        pixel_window_size = max(pixel_window_size - (int(downsample_factor) - 1), 1)
    return pixel_window_size


def select_max_response_audio_window_with_bounds(
    segment: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if segment.dim() != 4:
        raise ValueError(f"Expected segment shape [B, C, T, F], got {tuple(segment.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    num_time_steps = segment.shape[2]
    if num_time_steps <= 0:
        raise ValueError("Cannot select from an empty audio segment")

    scan_stride = max(1, window_size // 4)
    offsets = torch.arange(window_size, device=segment.device)
    max_start_idx = num_time_steps - window_size if num_time_steps >= window_size else num_time_steps - 1
    candidate_start_indices = list(range(0, max_start_idx + 1, scan_stride))
    if candidate_start_indices[-1] != max_start_idx:
        candidate_start_indices.append(max_start_idx)

    candidate_windows = []
    candidate_scores = []
    candidate_start_indices_tensor = torch.tensor(candidate_start_indices, device=segment.device, dtype=torch.long)
    for start_idx in candidate_start_indices:
        gather_indices = (start_idx + offsets).clamp(0, num_time_steps - 1).long()
        window = segment.index_select(dim=2, index=gather_indices)
        candidate_windows.append(window)
        candidate_scores.append(window.float().exp().sum(dim=(1, 2, 3)))

    scores = torch.stack(candidate_scores, dim=1)
    best_window_indices = scores.argmax(dim=1)
    best_start_indices = candidate_start_indices_tensor[best_window_indices]
    best_end_indices = torch.clamp(best_start_indices + window_size - 1, max=num_time_steps - 1)
    selected_windows = torch.cat(
        [
            candidate_windows[int(best_window_indices[batch_index])][batch_index : batch_index + 1]
            for batch_index in range(segment.shape[0])
        ],
        dim=0,
    )
    return selected_windows, best_start_indices, best_end_indices


def select_random_audio_window_with_bounds(
    segment: Tensor,
    window_size: int,
    *,
    rng: random.Random | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if segment.dim() != 4:
        raise ValueError(f"Expected segment shape [B, C, T, F], got {tuple(segment.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    batch_size = segment.shape[0]
    num_time_steps = segment.shape[2]
    if num_time_steps <= 0:
        raise ValueError("Cannot select from an empty audio segment")

    if num_time_steps <= window_size:
        start_indices = torch.zeros(batch_size, device=segment.device, dtype=torch.long)
    else:
        max_start_idx = num_time_steps - window_size
        if rng is None:
            start_indices = torch.randint(
                low=0,
                high=max_start_idx + 1,
                size=(batch_size,),
                device=segment.device,
            )
        else:
            start_indices = torch.tensor(
                [rng.randint(0, max_start_idx) for _ in range(batch_size)],
                device=segment.device,
                dtype=torch.long,
            )

    offsets = torch.arange(window_size, device=segment.device, dtype=torch.long)
    selected_windows = torch.cat(
        [
            segment[
                batch_index : batch_index + 1,
                :,
                (start_indices[batch_index] + offsets).clamp(0, num_time_steps - 1),
                :,
            ]
            for batch_index in range(batch_size)
        ],
        dim=0,
    )
    end_indices = torch.clamp(start_indices + window_size - 1, max=num_time_steps - 1)
    return selected_windows, start_indices, end_indices


def select_audio_window_with_bounds(
    segment: Tensor,
    window_size: int,
    *,
    mode: Literal["max_response", "random"] = "random",
    rng: random.Random | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if mode == "max_response":
        return select_max_response_audio_window_with_bounds(segment, window_size)
    if mode == "random":
        return select_random_audio_window_with_bounds(segment, window_size, rng=rng)
    raise ValueError(f"Unsupported audio window selection mode: {mode}")


def mel_window_bounds_to_seconds(
    start_index: int,
    end_index: int,
    *,
    hop_length: int,
    sample_rate: int,
) -> tuple[float, float]:
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    if end_index < start_index:
        raise ValueError(f"end_index must be >= start_index, got start={start_index}, end={end_index}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    start_time_sec = float(start_index * hop_length) / float(sample_rate)
    end_time_sec = float((end_index + 1) * hop_length) / float(sample_rate)
    return start_time_sec, end_time_sec


def select_video_frame_indices_from_time_range(
    *,
    num_frames: int,
    fps: float,
    start_time_sec: float,
    end_time_sec: float,
    count: int = 1,
    mode: Literal["first", "random", "center"] = "center",
    rng: random.Random | None = None,
) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if end_time_sec < start_time_sec:
        raise ValueError(f"end_time_sec must be >= start_time_sec, got {start_time_sec}, {end_time_sec}")

    mode = mode.lower()
    if mode not in {"first", "random", "center"}:
        raise ValueError(f"Unsupported frame selection mode: {mode}")

    start_frame = int(math.ceil(start_time_sec * fps))
    end_frame = int(math.ceil(end_time_sec * fps)) - 1
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(0, min(end_frame, num_frames - 1))

    if end_frame < start_frame:
        center_time_sec = max(0.0, 0.5 * (start_time_sec + end_time_sec))
        center_frame = int(round(center_time_sec * fps))
        candidate_frames = [max(0, min(center_frame, num_frames - 1))]
    else:
        candidate_frames = list(range(start_frame, end_frame + 1))

    if mode == "first":
        selected = candidate_frames[:count]
    elif mode == "center":
        if len(candidate_frames) <= count:
            selected = candidate_frames[:]
        else:
            center_offset = max(0, (len(candidate_frames) - count) // 2)
            selected = candidate_frames[center_offset : center_offset + count]
    else:
        rng = rng or random
        selected = (
            candidate_frames[:]
            if len(candidate_frames) <= count
            else sorted(rng.sample(candidate_frames, count))
        )

    if len(selected) < count:
        selected.extend([selected[-1]] * (count - len(selected)))
    return selected
