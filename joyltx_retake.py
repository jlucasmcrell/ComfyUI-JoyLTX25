"""JoyLTX Retake - regenerate one stretch of a finished clip and keep the rest.

Feed it a rendered clip (its frames and its audio) and a time window. Everything outside the
window is frozen as raw latents; only the window is denoised, from a prompt you write for that
moment. The two modalities are independent, which is where most of the value is:

  video + audio  - redo the moment completely.
  video only     - keep the performance, change what the camera sees. The voice, the timing and
                   the room tone are untouched, so it still cuts together.
  audio only     - keep the picture, change the line. Lip movement is whatever was rendered, so
                   keep the new line close in length to the old one. KNOWN LIMIT (blind-reviewed
                   2026-08-20): the voice inside the window is re-rolled from the prompt - it can
                   come back as a different speaker. Describe the voice precisely in the prompt,
                   and reroll seeds until it matches; a voice-carry lever is future work.

This is the inverse of the AV-extend join in the multishot sampler: there we freeze a head and
paint the rest, here we freeze everything except a window. Same raw-latent pinning, same masks.
"""
import time

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers

from nodes import VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import (SamplerCustomAdvanced, RandomNoise, ManualSigmas,
                                               KSamplerSelect)
from comfy_extras.nodes_lt import (LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
                                   LTXVDualCFGGuider)
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode, LTXVAudioVAEDecode

DIST_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
MODES = ["video + audio (redo the moment)",
         "video only (keep the performance)",
         "audio only (keep the picture)"]


def _first(out):
    a = getattr(out, "args", out)
    return a[0] if isinstance(a, (tuple, list)) else a


class JoyLTX_Retake:
    """Regenerate a time window of an existing clip (Joy-LTX 2.5)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "images": ("IMAGE", {"tooltip": "The finished clip's frames, in order."}),
            "audio": ("AUDIO", {"tooltip": "That clip's audio. Must be the same take as the frames."}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip":
                       "What should happen in the window. Write it as a shot prompt - the model only sees "
                       "this text plus the frozen material either side."}),
            "negative": ("STRING", {"multiline": True, "default":
                         "pc game, console game, video game, cartoon, childish, ugly"}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip":
                              "Where the retake starts. Snapped down to the latent grid (8 frames, 1/3 s at 24 fps)."}),
            "end_seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 600.0, "step": 0.1, "tooltip":
                            "Where it ends. Snapped up to the grid. A window under about a second has too "
                            "little room to differ from what is already there."}),
            "mode": (MODES, {"default": MODES[0]}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
            "sigmas": ("STRING", {"default": DIST_SIGMAS, "tooltip": "distilled 8-step schedule"}),
            "video_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "audio_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Redo one stretch of a finished clip - picture, sound, or both - and keep the rest."

    @staticmethod
    def _window(n_latent, n_pixels, start_s, end_s, fps):
        """Latent index range covering [start_s, end_s), clamped, at least one slot wide."""
        per = n_pixels / float(max(1, n_latent))          # pixel frames per latent slot
        a = int(max(0.0, start_s) * fps / per)
        b = int(round(min(end_s, n_pixels / fps) * fps / per + 0.5))
        a = max(0, min(a, n_latent - 1))
        b = max(a + 1, min(b, n_latent))
        return a, b

    def run(self, model, clip, video_vae, audio_vae, images, audio, prompt, negative,
            start_seconds, end_seconds, mode, seed, sampler_name, sigmas, video_cfg, audio_cfg,
            frame_rate):
        if end_seconds <= start_seconds:
            raise ValueError("JoyLTX Retake: end_seconds must be after start_seconds.")
        t0 = time.time()
        do_video = not mode.startswith("audio only")
        do_audio = not mode.startswith("video only")

        n_px = images.shape[0]
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative))
        pos, neg = LTXVConditioning.execute(pos, neg, frame_rate).args

        vlat = {"samples": video_vae.encode(images[:, :, :, :3])}
        alat = _first(LTXVAudioVAEEncode.execute(audio, audio_vae))
        av = _first(LTXVConcatAVLatent.execute(vlat, alat))
        v, a = av["samples"].unbind()

        # mask 1 = the sampler may paint here, 0 = frozen to what is already in the latent
        mv, ma = torch.zeros_like(v), torch.zeros_like(a)
        vw = aw = None
        if do_video:
            i, j = self._window(v.shape[2], n_px, start_seconds, end_seconds, frame_rate)
            mv[:, :, i:j] = 1.0
            vw = (i, j, v.shape[2])
        if do_audio:
            i, j = self._window(a.shape[2], n_px, start_seconds, end_seconds, frame_rate)
            ma[:, :, i:j] = 1.0
            aw = (i, j, a.shape[2])

        av = dict(av)
        av["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))

        sampler = _first(KSamplerSelect.execute(sampler_name))
        sig = _first(ManualSigmas.execute(sigmas))
        guider = _first(LTXVDualCFGGuider.execute(model, pos, neg, video_cfg, audio_cfg))
        out = _first(SamplerCustomAdvanced.execute(_first(RandomNoise.execute(seed)), guider,
                                                   sampler, sig, av))

        vfin, afin = LTXVSeparateAVLatent.execute(out).args
        dec = VAEDecodeTiled().decode(video_vae, vfin, 512, 64, 64, 16)
        imgs = dec[0] if isinstance(dec, (tuple, list)) else _first(dec)
        aud = _first(LTXVAudioVAEDecode.execute(afin, audio_vae))
        pk = float(aud["waveform"].abs().max())
        if pk > 0.99:                       # same decode limiter as the multishot sampler
            aud = dict(aud); aud["waveform"] = aud["waveform"] * (0.98 / pk)
            print("[JoyLTX Retake] audio peak %.3f > FS - limited to 0.98" % pk, flush=True)

        info = ("retake %.1f-%.1f s of a %.1f s clip | %s | video slots %s | audio slots %s | %.0f s"
                % (start_seconds, end_seconds, n_px / frame_rate, mode,
                   ("%d-%d of %d" % vw) if vw else "frozen",
                   ("%d-%d of %d" % aw) if aw else "frozen", time.time() - t0))
        print("[JoyLTX Retake] " + info, flush=True)
        comfy.model_management.soft_empty_cache()
        return (imgs, aud, info)


NODE_CLASS_MAPPINGS = {"JoyLTX_Retake": JoyLTX_Retake}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_Retake": "JoyLTX Retake (redo part of a clip)"}
