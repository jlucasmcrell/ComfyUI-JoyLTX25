"""LTX-2.5 multishot sampler (Joy-LTX 2.5): N shots from the writer, joined by AV-extend.

One node runs the whole two-pass LTX-2.5 pipeline once per shot and joins the shots:

  continue  - the previous shot's last `overlap` latent frames (video AND audio) are pinned
              at the head of the next shot (noise mask 0), the model paints the rest: one
              seamless take across generations, same voice, same room. The replayed head is
              trimmed on output.
  cut       - only the AUDIO tail is pinned: the voice carries straight across a picture cut,
              the picture is free (new angle / framing from the shot's prompt, optionally a
              first-frame image per shot).
  fresh     - nothing pinned; independent shots.

Built on the stock ComfyUI LTX nodes (Conditioning, EmptyLatentAudio, Concat/Separate AV,
DualCFG guider, SamplerCustomAdvanced, LatentUpsampler, tiled decode) - no new math, just the
loop and the masks. Prompts come in as the writer's JSON ({"prompts": [...]}) or a --- list.
"""
import json
import math
import os
import re
import time

import torch

import comfy.utils
import comfy.model_management
import comfy.samplers
import comfy.nested_tensor
import node_helpers
import folder_paths

from nodes import VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import (SamplerCustomAdvanced, RandomNoise, ManualSigmas,
                                                KSamplerSelect)
from comfy_extras.nodes_lt import (LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
                                   LTXVDualCFGGuider, LTXVImgToVideo, LTXVAddGuide, LTXVCropGuides)
from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio, LTXVAudioVAEDecode
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

_BLOCK_SPLIT = re.compile(r"(?m)^---\s*$")
DIST_SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DIST_SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"


def _first(out):
    """V3 nodes return io.NodeOutput; old nodes return tuples."""
    a = getattr(out, "args", out)
    return a[0] if isinstance(a, (tuple, list)) else a


def _parse_prompts(text):
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("prompts") or data.get("shots") or []
            if isinstance(data, list):
                out = []
                for item in data:
                    if isinstance(item, dict):
                        item = item.get("prompt") or item.get("text") or ""
                    if str(item).strip():
                        out.append(str(item).strip())
                if out:
                    return out
        except Exception:
            pass
    parts = [p.strip() for p in _BLOCK_SPLIT.split(text) if p.strip()]
    return parts if parts else [text]


class JoyLTX_Multishot:
    """LTX-2.5 Multishot Sampler (Joy-LTX 2.5)."""

    JOINS = ["continue (AV extend: seamless take)", "cut (voice extends, new picture)", "fresh (independent shots)"]
    IDREF = ["off", "cuts only: frame from shot 1", "all shots: frame from shot 1", "cuts only: last frame of previous shot"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompts": ("STRING", {"multiline": True, "default": "", "tooltip":
                        "The writer's shot prompts: {\"prompts\": [...]} JSON or blocks separated by --- "
                        "(wire the writer here, or paste your own)."}),
            "negative": ("STRING", {"multiline": True, "default":
                         "pc game, console game, video game, cartoon, childish, ugly"}),
            "width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 32}),
            "height": ("INT", {"default": 544, "min": 256, "max": 1920, "step": 32}),
            "frames_per_shot": ("INT", {"default": 193, "min": 25, "max": 1441, "step": 8, "tooltip":
                                "8n+1 frames per shot at 24 fps (193 = 8 s)."}),
            "shot_count": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip":
                           "0 = every prompt the writer produced; N = the first N."}),
            "join": (cls.JOINS, {"default": cls.JOINS[0]}),
            "overlap": ("INT", {"default": 3, "min": 1, "max": 12, "tooltip":
                        "Latent frames of the previous shot pinned at the head of the next one "
                        "(3 = 17 pixel frames = 0.7 s). More = smoother join, less new content per shot."}),
            "seed": ("INT", {"default": 553010, "min": 0, "max": 0xffffffffffffffff}),
            "seed_per_shot": ("BOOLEAN", {"default": True}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
            "sigmas_pass1": ("STRING", {"default": DIST_SIGMAS_1, "tooltip": "distilled 8-step schedule"}),
            "two_pass": ("BOOLEAN", {"default": True, "tooltip":
                         "Upscale each shot with the latent upsampler and refine (needs upscale_model)."}),
            "sigmas_pass2": ("STRING", {"default": DIST_SIGMAS_2}),
            "video_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "audio_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 12.0, "max": 60.0, "step": 1.0}),
            "save_every_shot": ("BOOLEAN", {"default": False, "tooltip":
                                "Also write each shot (untrimmed) as output/video/LTX_SHOTS/shot_*.mp4"}),
            "identity_ref": (cls.IDREF, {"default": cls.IDREF[1], "tooltip":
                             "Keep the SAME people across shots without reference images: a frame of shot 1 "
                             "(or the previous shot's last frame) is attached to every later shot as an in-context "
                             "keyframe reference (LTX-2.5 AddGuide, appended tokens, cropped after sampling). "
                             "'cuts only' = cut/fresh modes; 'all shots' = also in continue mode (anchors the look "
                             "to shot 1, which also fights slow texture drift on long takes)."}),
            "identity_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                                  "How hard the reference pulls (attention weight of the appended keyframe). "
                                  "0.4-0.7: same face/clothes/room, free composition; 1.0: near first-frame copy."}),
        }, "optional": {
            "upscale_model": ("LATENT_UPSCALE_MODEL",),
            "start_image": ("IMAGE", {"tooltip": "First frame of shot 1 (image-to-video)."}),
            "shot_images": ("IMAGE", {"tooltip":
                            "One image per shot (batch); used as the first frame of each shot in "
                            "cut/fresh mode (identity carry from your reference plates)."}),
            "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "end_images": ("IMAGE", {"tooltip": "One image per shot = keyframe at that shot's LAST frame "
                                     "(first->last-frame shots when combined with shot_images / identity)."}),
            "keyframe_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "How hard the end keyframes pull."}),
            "ref_images": ("IMAGE", {"tooltip": "One REFERENCE PHOTO per shot (from JoyLTX Refs by Name, or any batch): "
                                     "attached to that shot as an in-context keyframe at frame 0 (appended tokens, cropped "
                                     "after pass 1) so the person in the photo is the person in the shot. Replaces the "
                                     "frame-of-shot-1 identity for shots that have a photo."}),
            "ref_mask": ("STRING", {"default": "", "tooltip": "Comma list, one token per shot from Refs by Name: the "
                                    "CHARACTER NAME of that shot ('-' = none). With names, the sampler locks each character to "
                                    "their own first rendered frame (visual lock) and their own audio tail (voice lock) in later "
                                    "shots. 1/0 also accepted (photo / no photo)."}),
            "ref_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "How hard a reference photo pulls (measured on one seed, 0.5/0.75/0.9/1.0). "
                             "Below ~0.75 you get the hair and the clothes but a different face; 0.85-0.9 carries "
                             "the face and the small things like glasses; 1.0 drags the photo's own room into the "
                             "shot. Note the photograph does NOT carry age - the prompt has to say it."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = "JoyLTX"
    DESCRIPTION = ("Runs the LTX-2.5 two-pass pipeline once per shot and joins the shots with an "
                   "AV-extend (previous tail pinned as raw latents), so a take can be as long as "
                   "you like and a cut keeps the voice.")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode(clip, text):
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    @staticmethod
    def _sample(model, positive, negative, latent, sigmas, sampler, seed, video_cfg, audio_cfg):
        guider = _first(LTXVDualCFGGuider.execute(model, positive, negative, video_cfg, audio_cfg))
        noise = _first(RandomNoise.execute(seed))
        out = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)
        return _first(out)

    @staticmethod
    def _tail(av_latent, k_video):
        """Last k video latent frames + the matching audio tail (both raw)."""
        v, a = av_latent["samples"].unbind()
        v = v[:, :, -k_video:].clone().cpu()
        # audio latent (b, c, T, f): take the tail proportional to the pinned pixel span
        n_v = av_latent["samples"].unbind()[0].shape[2]
        k_a = max(1, int(round(a.shape[2] * (k_video / float(n_v)))))
        a = a[:, :, -k_a:].clone().cpu()
        return v, a

    @staticmethod
    def _pin(av, tail, mode, atail=None, pin_audio=True):
        """Pin the previous shot's tail (video+audio raw latents) at the head of `av` (noise mask 0).

        `atail` pins the AUDIO from a different tail than the picture - used when the speaker
        changes but the take must stay seamless: the picture continues from the previous shot,
        the voice comes from that speaker's own earlier shot. `pin_audio=False` leaves the audio
        head free, so a speaker heard for the first time gets their own voice instead of
        continuing the last one.
        """
        v, a = av["samples"].unbind()
        v = v.clone(); a = a.clone()
        if "noise_mask" in av:
            mv, ma = av["noise_mask"].unbind()
            mv = mv.clone(); ma = ma.clone()
        else:
            mv, ma = torch.ones_like(v), torch.ones_like(a)
        pv, pa = tail
        if atail is not None:
            pa = atail[1]
        if mode == "continue" and pv.shape[-2:] == v.shape[-2:]:
            kv = min(pv.shape[2], v.shape[2] - 1)
            v[:, :, :kv] = pv[:, :, -kv:].to(v.device, v.dtype)
            mv[:, :, :kv] = 0.0
        if pin_audio:
            ka = min(pa.shape[2], a.shape[2] - 1)
            a[:, :, :ka] = pa[:, :, -ka:].to(a.device, a.dtype)
            ma[:, :, :ka] = 0.0
        out = dict(av)
        out["samples"] = comfy.nested_tensor.NestedTensor((v, a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))
        return out

    # ------------------------------------------------------------------ main
    def run(self, model, clip, video_vae, audio_vae, prompts, negative, width, height, frames_per_shot,
            shot_count, join, overlap, seed, seed_per_shot, sampler_name, sigmas_pass1, two_pass,
            sigmas_pass2, video_cfg, audio_cfg, frame_rate, save_every_shot,
            identity_ref="cuts only: frame from shot 1", identity_strength=0.6,
            upscale_model=None, start_image=None, shot_images=None, image_strength=1.0,
            end_images=None, keyframe_strength=0.8, ref_images=None, ref_mask="", ref_strength=0.5):
        t0 = time.time()
        shots = _parse_prompts(prompts)
        if not shots:
            raise ValueError(
                "JoyLTX Multishot: the prompts input is empty. If the node's `prompts` slot is not connected, this "
                "canvas is a STALE copy in the browser - close the tab and open the workflow again from the sidebar "
                "(ComfyUI drops links when a node's inputs change on disk). Otherwise check the writer / Prompts from "
                "File node upstream is not bypassed or muted.")
        if shot_count > 0:
            shots = shots[:shot_count]
        n = len(shots)
        frames = ((frames_per_shot - 1) // 8) * 8 + 1
        w, h = (width // 32) * 32, (height // 32) * 32
        mode = "continue" if join.startswith("continue") else ("cut" if join.startswith("cut") else "fresh")
        two_pass = bool(two_pass and upscale_model is not None)
        sampler = _first(KSamplerSelect.execute(sampler_name))
        sig1 = _first(ManualSigmas.execute(sigmas_pass1))
        sig2 = _first(ManualSigmas.execute(sigmas_pass2))
        neg = self._encode(clip, negative)
        head_px = 1 + 8 * (overlap - 1)          # pixel frames the pinned latents decode to
        print(f"[JoyLTX Multishot] {n} shot(s) x {frames}f @ {w}x{h} | join={mode} overlap={overlap} "
              f"latent frames (trim {head_px}px on shots 2+) | two_pass={two_pass}", flush=True)

        images_out, audio_out, sr = [], [], None
        prev_tail, prev_tail2 = None, None
        anchor_tail, anchor_tail2 = None, None   # shot 1's tails - the fixed voice anchor
        # Distinct fixed voice sentences across the piece. At most one means the
        # whole piece is one voice, so the audio head can be anchored to shot 1
        # instead of walking shot-to-shot.
        _voices = {m.group(0) for s in shots for m in re.finditer(r"voice is[^.]*\.", s)}
        single_voice = len(_voices) <= 1
        prev_cname = None                 # who spoke in the last shot - decides where the next voice is pinned from
        char_frame, char_tail, char_tail2 = {}, {}, {}
        info = []
        for i, text in enumerate(shots):
            s_seed = seed + i if seed_per_shot else seed
            pos = self._encode(clip, text)
            pos, negc = LTXVConditioning.execute(pos, neg, frame_rate).args
            # ---- latents for this shot
            first_img = None
            if i == 0 and start_image is not None:
                first_img = start_image[:1]
            elif shot_images is not None and mode != "continue":
                first_img = shot_images[min(i, shot_images.shape[0] - 1):min(i, shot_images.shape[0] - 1) + 1]
            if first_img is not None:
                pos, negc, vlat = LTXVImgToVideo.execute(pos, negc, first_img, video_vae, w, h, frames, 1,
                                                         image_strength).args
            else:
                vlat = {"samples": torch.zeros([1, 128, (frames - 1) // 8 + 1, h // 32, w // 32],
                                               device=comfy.model_management.intermediate_device())}
            # ---- reference PHOTO for this shot (Refs by Name) + per-CHARACTER identity lock
            #      ref_mask = one token per shot: a character name (from Refs by Name), or 1/0 (legacy).
            #      First appearance of a character: their photo at ref_strength.
            #      Later shots of the same character: their OWN rendered frame from their first shot at identity_strength
            #      (the visual lock) + the photo at half ref_strength; and in cut mode their own audio tail is pinned
            #      instead of the previous shot's (the voice lock), so alternating speakers keep their voices.
            has_guide = False
            used_photo = False
            mk = [m.strip() for m in str(ref_mask or "").split(",")]
            tok = mk[i] if i < len(mk) else (mk[-1] if mk else "")
            cname = None
            if tok and tok not in ("0", "1", "-"):
                cname = tok.lower()
            has_photo = bool(tok) and tok not in ("0", "-") if mk and any(mk) else (ref_images is not None)
            if ref_images is not None and ref_strength > 0 and first_img is None and has_photo and ref_images.shape[0] > 0:
                rimg = ref_images[min(i, ref_images.shape[0] - 1): min(i, ref_images.shape[0] - 1) + 1]
                if cname and cname in char_frame and identity_strength > 0:
                    pos, negc, vlat = LTXVAddGuide.execute(pos, negc, video_vae, vlat, char_frame[cname], 0, float(identity_strength)).args
                    pos, negc, vlat = LTXVAddGuide.execute(pos, negc, video_vae, vlat, rimg, 0, float(ref_strength) * 0.5).args
                else:
                    pos, negc, vlat = LTXVAddGuide.execute(pos, negc, video_vae, vlat, rimg, 0, float(ref_strength)).args
                has_guide = True; used_photo = True
            # ---- identity reference: a frame of an earlier shot as an in-context keyframe (appended, cropped later)
            if not used_photo and i > 0 and identity_ref != "off" and identity_strength > 0 and first_img is None:
                use = (identity_ref.startswith("all") or mode != "continue")
                if use:
                    if identity_ref.endswith("previous shot"):
                        ref = images_out[-1][-1:]
                    else:
                        ref = images_out[0][images_out[0].shape[0] // 2: images_out[0].shape[0] // 2 + 1]
                    pos, negc, vlat = LTXVAddGuide.execute(pos, negc, video_vae, vlat, ref, 0, float(identity_strength)).args
                    has_guide = True
            # ---- end keyframe for this shot (LTX-2.5 keyframe at the last frame; cropped after pass 1)
            if end_images is not None and keyframe_strength > 0:
                e = end_images[min(i, end_images.shape[0] - 1): min(i, end_images.shape[0] - 1) + 1]
                pos, negc, vlat = LTXVAddGuide.execute(pos, negc, video_vae, vlat, e, -1, float(keyframe_strength)).args
                has_guide = True
            alat = _first(LTXVEmptyLatentAudio.execute(frames, frame_rate, 1, audio_vae))
            av = _first(LTXVConcatAVLatent.execute(vlat, alat))
            # ---- pin the previous tail (AV extend), pass-1 grid.
            #      The pinned audio head decides the VOICE, so who speaks next decides where it comes from:
            #        same speaker as the last shot -> the previous tail, the voice simply continues;
            #        a speaker heard before        -> their OWN last tail (cut mode swaps the whole pin,
            #                                        continue mode keeps the picture seam and swaps only the audio);
            #        a speaker heard for the first time -> no audio pin at all, or they inherit the last voice.
            pin_src, pin_src2 = prev_tail, prev_tail2
            voice1 = voice2 = None
            pin_audio = True
            if cname and cname != prev_cname:
                if cname in char_tail:
                    if mode != "continue":
                        pin_src, pin_src2 = char_tail[cname], char_tail2.get(cname)
                    else:
                        voice1, voice2 = char_tail[cname], char_tail2.get(cname)
                    print("[JoyLTX Multishot] shot %d: voice for '%s' from their own earlier shot" % (i + 1, cname),
                          flush=True)
                elif prev_tail is not None:
                    pin_audio = False
                    print("[JoyLTX Multishot] shot %d: '%s' speaks for the first time - audio head left free so "
                          "they do not inherit the previous voice" % (i + 1, cname), flush=True)
            # ---- VOICE ANCHOR (cut mode). Pinning prev_tail makes each shot
            #      imitate the previous shot's imitation - by shot 6 the voice
            #      has walked (reported 2026-08-21: single unnamed speaker, cut
            #      mode, identity gone by the tail of the piece). Anchor the
            #      audio head to a FIXED tail instead: a named speaker to their
            #      own first shot (every shot, not only on speaker changes),
            #      an unnamed single-voice piece to shot 1. Safe in cut mode
            #      because _pin copies the VIDEO half only under continue -
            #      this swap touches nothing but the voice. Continue mode is
            #      left alone: its audio genuinely continues through the seam.
            #      JOYLTX_VOICE_WALK=1 restores the old behaviour for A/B.
            if mode == "cut" and not os.environ.get("JOYLTX_VOICE_WALK"):
                if cname and cname in char_tail:
                    pin_src, pin_src2 = char_tail[cname], char_tail2.get(cname)
                elif cname is None and single_voice and anchor_tail is not None:
                    pin_src = anchor_tail
                    pin_src2 = anchor_tail2 if anchor_tail2 is not None else pin_src2
                    if i == 1:
                        print("[JoyLTX Multishot] single-voice piece: audio head anchored to "
                              "shot 1's tail for every later shot (JOYLTX_VOICE_WALK=1 reverts)",
                              flush=True)
            elif mode == "continue" and not os.environ.get("JOYLTX_VOICE_WALK"):
                # Same anchor for seamless takes, through the mechanism the
                # speaker-change path already uses: swap ONLY the audio half of
                # the pin (atail) - the picture seam still continues from the
                # previous shot, which is what makes the take seamless. Scoping
                # the 2026-08-21 anchor to cut mode assumed continue mode's
                # audio "genuinely flows"; field result the same day: same-
                # speaker shots still walked. The pinned head is trimmed on
                # decode, so it conditions the voice without being heard.
                if cname and cname in char_tail:
                    voice1, voice2 = char_tail[cname], char_tail2.get(cname)
                elif cname is None and single_voice and anchor_tail is not None:
                    voice1 = anchor_tail
                    voice2 = anchor_tail2
                    if i == 1:
                        print("[JoyLTX Multishot] seamless take, single voice: audio "
                              "head anchored to shot 1 every shot; picture seam "
                              "untouched (JOYLTX_VOICE_WALK=1 reverts)", flush=True)
            if pin_src is not None and mode != "fresh":
                av = self._pin(av, pin_src, mode, voice1, pin_audio)
            # ---- pass 1
            ts = time.time()
            out1 = self._sample(model, pos, negc, av, sig1, sampler, s_seed, video_cfg, audio_cfg)
            v1, a1 = LTXVSeparateAVLatent.execute(out1).args
            if has_guide:
                pos, negc, v1 = LTXVCropGuides.execute(pos, negc, v1).args   # drop the appended reference tokens
            v1 = {"samples": v1["samples"]}; a1 = {"samples": a1["samples"]}   # no stale masks into pass 2
            out1 = _first(LTXVConcatAVLatent.execute(v1, a1))
            prev_tail = self._tail(out1, overlap)      # pin from PASS-1 latents (same grid as the next shot)
            if anchor_tail is None:
                anchor_tail = prev_tail                # shot 1's tail = the piece's voice anchor
            final = out1
            # ---- pass 2 (upscale + refine)
            if two_pass:
                up = _first(LTXVLatentUpsampler.execute(v1, upscale_model, video_vae))
                av2 = _first(LTXVConcatAVLatent.execute(up, a1))
                # pin the previous shot's REFINED tail too, so pass 2 does not re-draw the join
                if pin_src2 is not None and mode != "fresh":
                    av2 = self._pin(av2, pin_src2, mode, voice2, pin_audio)
                final = self._sample(model, pos, negc, av2, sig2, sampler, s_seed, video_cfg, audio_cfg)
                prev_tail2 = self._tail(final, overlap)
                if anchor_tail2 is None:
                    anchor_tail2 = prev_tail2
            # ---- decode
            vfin, afin = LTXVSeparateAVLatent.execute(final).args
            dec = VAEDecodeTiled().decode(video_vae, vfin, 512, 64, 64, 16)
            imgs = dec[0] if isinstance(dec, (tuple, list)) else _first(dec)
            aud = _first(LTXVAudioVAEDecode.execute(afin, audio_vae))
            sr = aud["sample_rate"]
            wav = aud["waveform"]
            if save_every_shot:
                self._save_shot(imgs, aud, frame_rate, i)
            # ---- trim the replayed head on shots 2+
            if i > 0 and mode != "fresh":
                imgs = imgs[head_px:]
                cut = int(round(head_px / frame_rate * sr))
                wav = wav[..., cut:]
            # keep the audio exactly as long as the video
            want = int(round(imgs.shape[0] / frame_rate * sr))
            if wav.shape[-1] > want:
                wav = wav[..., :want]
            elif wav.shape[-1] < want:
                wav = torch.nn.functional.pad(wav, (0, want - wav.shape[-1]))
            images_out.append(imgs.cpu())
            audio_out.append(wav.cpu())
            if cname and cname not in char_frame:      # first appearance: remember this character's rendered face + voice
                mid = imgs.shape[0] // 2
                fr = imgs[mid:mid + 1].cpu()
                # anchor on the PERSON, not the room: central upper 60 % of the frame, scaled back up
                H_, W_ = fr.shape[1], fr.shape[2]
                ch_, cw_ = int(H_ * 0.6), int(W_ * 0.6)
                top_ = int((H_ - ch_) * 0.25); left_ = (W_ - cw_) // 2
                crop_ = fr[:, top_:top_ + ch_, left_:left_ + cw_, :]
                crop_ = torch.nn.functional.interpolate(crop_.movedim(-1, 1), size=(H_, W_), mode="bilinear", align_corners=False).movedim(1, -1)
                char_frame[cname] = crop_.contiguous()
                char_tail[cname] = prev_tail
                char_tail2[cname] = prev_tail2
                print(f"[JoyLTX Multishot] identity+voice lock set for '{cname}' from shot {i+1}", flush=True)
            prev_cname = cname
            info.append(f"shot {i+1}/{n}: {imgs.shape[0]}f in {time.time()-ts:.0f}s")
            print(f"[JoyLTX Multishot] {info[-1]}", flush=True)
            comfy.model_management.soft_empty_cache()

        images = torch.cat(images_out, dim=0)
        wav_all = torch.cat(audio_out, dim=-1)
        # DECODE LIMITER. The audio VAE routinely decodes past full scale (measured 1.13-1.36),
        # and everything downstream - the mux, SaveAudio, any player - hard-clips it, which is
        # the "frying" on loud breaths and voices. Normalised once here, on the tensor this node
        # actually returns, so no later trim/pad path can slip past it. -0.3 dBFS leaves headroom
        # for the lossy encoder's inter-sample overshoot.
        pk_all = float(wav_all.abs().max())
        if pk_all > 0.97:
            wav_all = wav_all * (0.97 / pk_all)
        print("[JoyLTX Multishot] audio peak %.3f%s" % (pk_all,
              " -> limited to 0.97 (was over full scale)" if pk_all > 0.97 else " (under FS, untouched)"),
              flush=True)
        audio = {"waveform": wav_all, "sample_rate": sr}
        total = f"{n} shots -> {images.shape[0]} frames (~{images.shape[0]/frame_rate:.1f}s) in {time.time()-t0:.0f}s"
        print(f"[JoyLTX Multishot] done: {total}", flush=True)
        return (images, audio, total + "\n" + "\n".join(info))

    @staticmethod
    def _save_shot(images, audio, fps, idx):
        try:
            import os
            from comfy_extras.nodes_video import CreateVideo
            vid = _first(CreateVideo.execute(images, float(fps), audio))
            folder = os.path.join(folder_paths.get_output_directory(), "video", "JOYLTX_SHOTS")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "shot_%02d_%d.mp4" % (idx + 1, int(time.time())))
            vid.save_to(path)
            print(f"[JoyLTX Multishot] shot {idx+1} saved -> {path}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[JoyLTX Multishot] per-shot save skipped ({e})", flush=True)


NODE_CLASS_MAPPINGS = {"JoyLTX_Multishot": JoyLTX_Multishot}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_Multishot": "JoyLTX Multishot Sampler (LTX-2.5, AV-extend joins)"}
