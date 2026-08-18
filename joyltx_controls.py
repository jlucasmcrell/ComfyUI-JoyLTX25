"""Joy-LTX 2.5 controls: the take planner (VRAM-aware pass planning) and the lazy A/B switch."""
import json
import math
import re

class JoyLTX_TakeControls:
    """MASTER CONTROLS for the LTX-2.5 single-generation canvas (Joy-LTX 2.5).

    LTX renders the whole take in ONE generation. This panel does for LTX what
    plan_take does for H3: it sizes the render to the card. LTX's cost model is
    tokens = (W/32)(H/32) x latent frames, and pass 2 (the upscaled refine)
    runs on the OUTPUT grid - so the upscale factor decides how much a take
    costs: x2 = 4x the pixels of pass 1, x1.5 = 2.25x, none = pass 1 only.
    Measured 2026-08-17: 960x544 -> x2 -> 1920x1088 at 193 frames (51k pass-2
    tokens) renders on 24 GB; 481 frames (124k) hangs a 32 GB card fully
    offloaded. So `auto` keeps the render size and picks the largest output
    that fits: x2, then x1.5 (the LTX-2.3 x1.5 spatial upscaler, verified to
    work on 2.5 latents), then a single pass. It prints the plan.
    """

    TOKENS_PER_GB = 2200      # pass-2 tokens per GB of card (measured, see docstring)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 960, "min": 256, "max": 1920, "step": 32,
                "tooltip": "RENDER (pass-1) width. Output = this x the upscale factor "
                "(960x544 x2 = 1920x1088, x1.5 = 1440x832). Multiples of 32."}),
            "height": ("INT", {"default": 544, "min": 256, "max": 1920, "step": 32,
                "tooltip": "Render (pass-1) height; output = this x the upscale factor."}),
            "take_seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Length of the take, rendered in ONE generation. Longer takes "
                "cost tokens; past what the card fits, auto steps the upscale down "
                "(x2 -> x1.5 -> none) and prints which. 8 s x2 / ~14 s x1.5 at "
                "960x544 is the 24 GB comfort zone."}),
            "beat_seconds": ("FLOAT", {"default": 8.0, "min": 3.0, "max": 15.0, "step": 0.5,
                "tooltip": "How long each written sentence/beat should be; the writer "
                "gets take_seconds / beat_seconds beats to write."}),
            "upscale": (["auto", "x2", "x1.5", "none"], {"default": "auto",
                "tooltip": "auto = the largest output that fits your card. x2 = the "
                "official LTX-2.5 spatial upscaler (4x pixels in pass 2). x1.5 = "
                "the LTX-2.3 spatial x1.5 upscaler file - works on 2.5 latents "
                "(verified render), 2.25x pixels, so about 1.8x the seconds of "
                "x2 on the same card. none = pass 1 only at the render size."}),
        }, "optional": {
            "vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 1.0,
                "tooltip": "0 = read the card. Set a number to plan for another card."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN", "BOOLEAN", "INT", "INT", "STRING")
    RETURN_NAMES = ("pass1_width", "pass1_height", "frames", "two_pass", "use_x15", "beats", "beat_frames", "summary")
    FUNCTION = "emit"
    CATEGORY = "JoyLTX"

    @staticmethod
    def _snap32(v):
        return max(256, int(round(v / 32.0)) * 32)

    @staticmethod
    def _up15(v):
        # the rational x1.5 head shuffles x3 then blur-downs /2 -> ceil on odd latent counts
        return int(math.ceil((v // 32) * 1.5)) * 32

    @staticmethod
    def _tokens(w, h, frames):
        return (w // 32) * (h // 32) * ((frames - 1) // 8 + 1)

    def emit(self, width, height, take_seconds, beat_seconds=8.0, upscale="auto", vram_gb=0.0):
        import math
        frames = int(round(float(take_seconds) * 24))
        frames = max(9, int(round((frames - 1) / 8.0)) * 8 + 1)
        beats = max(1, int(math.ceil(float(take_seconds) / float(beat_seconds))))
        beat_frames = max(9, int(round((float(beat_seconds) * 24 - 1) / 8.0)) * 8 + 1)
        total = float(vram_gb) if vram_gb and vram_gb > 0 else 0.0
        if total <= 0:
            try:
                import comfy.model_management as mm
                total = mm.get_total_memory(mm.get_torch_device()) / (1024 ** 3)
            except Exception:
                total = 24.0
        budget = self.TOKENS_PER_GB * total
        w, h = self._snap32(width), self._snap32(height)
        t1 = self._tokens(w, h, frames)
        t2 = self._tokens(2 * w, 2 * h, frames)
        w15, h15 = self._up15(w), self._up15(h)
        t15 = self._tokens(w15, h15, frames)
        two, use15 = False, False
        if upscale == "x2" or (upscale == "auto" and t2 <= budget):
            two = True
            why = "two passes x2 -> %dx%d (%.0fk pass-2 tokens" % (2 * w, 2 * h, t2 / 1e3)
            why += ", over the ~%.0fk this %.0f GB card fits - FORCED, expect streaming or an OOM)" % (budget / 1e3, total) if t2 > budget else ", of ~%.0fk this %.0f GB card fits)" % (budget / 1e3, total)
        elif upscale == "x1.5" or (upscale == "auto" and t15 <= budget):
            two, use15 = True, True
            why = "two passes x1.5 -> %dx%d (%.0fk pass-2 tokens" % (w15, h15, t15 / 1e3)
            why += ", over the ~%.0fk this %.0f GB card fits - FORCED, expect streaming or an OOM)" % (budget / 1e3, total) if t15 > budget else ", of ~%.0fk this %.0f GB card fits; x2 would need %.0fk)" % (budget / 1e3, total, t2 / 1e3)
        else:
            why = "ONE pass -> %dx%d (%.0fk tokens; x1.5 would need %.0fk, x2 %.0fk; ~%.0fk fits this %.0f GB card)" % (w, h, t1 / 1e3, t15 / 1e3, t2 / 1e3, budget / 1e3, total)
            if t1 > budget:
                why += " - even one pass is over budget: shorten the take or lower the size"
        summary = ("LTX TAKE: %.1f s -> %d frames in ONE generation | render %dx%d | %s | writer gets %d "
                   "beat(s) of %d frames" % (float(take_seconds), frames, w, h, why, beats, beat_frames))
        print("[JoyLTX TakeControls] " + summary, flush=True)
        return (w, h, frames, two, use15, beats, beat_frames, summary)


class _AnyT(str):
    """Wildcard socket type: compares unequal to nothing, so any link is accepted."""
    def __ne__(self, other):
        return False


_any = _AnyT("*")


class JoyLTX_AnySwitch:
    """Lazy A/B switch: only the branch the boolean selects is ever executed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"use_on": ("BOOLEAN", {"forceInput": True})},
            "optional": {"off_path": (_any, {"lazy": True}), "on_path": (_any, {"lazy": True})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (_any,)
    RETURN_NAMES = ("value",)
    FUNCTION = "pick"
    CATEGORY = "JoyLTX"

    def check_lazy_status(self, use_on, off_path=None, on_path=None, prompt=None, unique_id=None):
        return ["on_path" if use_on else "off_path"]

    def pick(self, use_on, off_path=None, on_path=None, prompt=None, unique_id=None):
        name = "on_path" if use_on else "off_path"
        v = on_path if use_on else off_path
        if v is None:
            wired = False
            try:
                wired = isinstance(prompt[str(unique_id)]["inputs"].get(name), list)
            except Exception:
                wired = False
            if not wired:
                raise ValueError("JoyLTX AnySwitch: the selected input (%s) is not connected" % name)
        return (v,)


class JoyLTX_OptionalImage:
    """Optional I2V image: passes the image through when ON, emits nothing when OFF (true T2V),
    and a BOOLEAN that drives the empty-latent / image-to-video switch on the Take canvas.
    The image input is lazy, so the Load Image upstream is skipped entirely when OFF."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"enabled": ("BOOLEAN", {"default": False, "label_on": "I2V: image ON",
                                                 "label_off": "T2V (no image)"})},
            "optional": {"image": ("IMAGE", {"lazy": True, "tooltip": "First frame when enabled."})},
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "enabled")
    FUNCTION = "gate"
    CATEGORY = "JoyLTX"

    def check_lazy_status(self, enabled, image=None):
        return ["image"] if enabled else []

    def gate(self, enabled, image=None):
        if not enabled or image is None:
            if enabled:
                print("[JoyLTX OptionalImage] enabled but no image connected - T2V.", flush=True)
            return (None, False)
        return (image, True)


class JoyLTX_Keyframes:
    """Optional LTX-2.5 keyframes for a take: an END frame (last frame of the take), and/or MID
    keyframes (an image batch with a comma list of frame indices). Images are appended as keyframe
    tokens (LTX-2.5 AddGuide) - they condition, they are cropped out after pass 1 by LTXVCropGuides.
    With no images connected the node passes everything through unchanged (no cost)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                "vae": ("VAE",), "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                             "tooltip": "How hard the keyframes pull (1.0 = the frame is reproduced)."}),
                "mid_frame_indices": ("STRING", {"default": "", "tooltip":
                                      "Comma list, one pixel-frame index per mid image, e.g. 48,96 (rounded to 8n)."}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "Last frame of the take."}),
                "mid_images": ("IMAGE", {"tooltip": "Batch of mid keyframes, in the order of mid_frame_indices."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "JoyLTX"

    def apply(self, positive, negative, vae, latent, strength, mid_frame_indices="", end_image=None, mid_images=None):
        if end_image is None and mid_images is None:
            return (positive, negative, latent)
        from comfy_extras.nodes_lt import LTXVAddGuide
        n = 0
        if mid_images is not None and str(mid_frame_indices).strip():
            idxs = [int(float(x)) for x in str(mid_frame_indices).replace(";", ",").split(",") if x.strip()]
            for k, fi in enumerate(idxs[:mid_images.shape[0]]):
                fi = max(0, (fi // 8) * 8)
                positive, negative, latent = LTXVAddGuide.execute(positive, negative, vae, latent, mid_images[k:k + 1], fi, float(strength)).args
                n += 1
        if end_image is not None:
            positive, negative, latent = LTXVAddGuide.execute(positive, negative, vae, latent, end_image[:1], -1, float(strength)).args
            n += 1
        print(f"[JoyLTX Keyframes] {n} keyframe(s) attached (strength {strength}); crop them after pass 1.", flush=True)
        return (positive, negative, latent)


class JoyLTX_ImageFolder:
    """Load a folder of images (sorted by name) as ONE batch - one image per shot for the multishot
    sampler's shot_images / end_images. Folder is relative to ComfyUI/input (or absolute). Images are
    resized to the first image's size. Empty / missing folder -> emits nothing (the input stays unset)."""

    @classmethod
    def _folders(cls):
        import os, folder_paths
        root = folder_paths.get_input_directory()
        out = ["(none)"]
        try:
            for dp, dns, fns in os.walk(root):
                rel = os.path.relpath(dp, root).replace(os.sep, "/")
                if rel == ".":
                    continue
                if any(f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) for f in fns):
                    out.append(rel)
        except Exception:
            pass
        return out

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": (cls._folders(), {"default": "(none)", "tooltip":
                       "A folder under ComfyUI/input holding ONE image per shot, named in shot order "
                       "(01.png, 02.png ...). Make the folder, drop the images in, press R to refresh the list."}),
            "enabled": ("BOOLEAN", {"default": False, "label_on": "load folder", "label_off": "off (emit nothing)"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load"
    CATEGORY = "JoyLTX"

    @classmethod
    def IS_CHANGED(cls, folder, enabled):
        import os, glob
        if not enabled or not folder.strip() or folder == "(none)":
            return "off"
        d = cls._resolve(folder)
        files = sorted(glob.glob(os.path.join(d, "*")))
        return str([(f, os.path.getmtime(f)) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])

    @staticmethod
    def _resolve(folder):
        import os, folder_paths
        f = folder.strip().strip('"')
        return f if os.path.isabs(f) else os.path.join(folder_paths.get_input_directory(), f)

    def load(self, folder, enabled):
        import os, glob
        import numpy as np, torch
        from PIL import Image, ImageOps
        if not enabled or not folder.strip() or folder == "(none)":
            return (None, 0)
        d = self._resolve(folder)
        files = [f for f in sorted(glob.glob(os.path.join(d, "*"))) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        if not files:
            print(f"[JoyLTX ImageFolder] no images in {d} - emitting nothing.", flush=True)
            return (None, 0)
        imgs, size = [], None
        for f in files:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            if size is None:
                size = im.size
            elif im.size != size:
                im = im.resize(size, Image.LANCZOS)
            imgs.append(torch.from_numpy(np.array(im).astype(np.float32) / 255.0)[None])
        batch = torch.cat(imgs, 0)
        print(f"[JoyLTX ImageFolder] {batch.shape[0]} image(s) from {d} at {size[0]}x{size[1]}", flush=True)
        return (batch, batch.shape[0])


NODE_CLASS_MAPPINGS = {"JoyLTX_TakeControls": JoyLTX_TakeControls, "JoyLTX_AnySwitch": JoyLTX_AnySwitch,
                       "JoyLTX_OptionalImage": JoyLTX_OptionalImage, "JoyLTX_Keyframes": JoyLTX_Keyframes,
                       "JoyLTX_ImageFolder": JoyLTX_ImageFolder}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_TakeControls": "JoyLTX Take Controls (render size / seconds / upscale)",
                              "JoyLTX_AnySwitch": "JoyLTX Any Switch (lazy A/B)",
                              "JoyLTX_OptionalImage": "JoyLTX Optional Image (I2V on/off)",
                              "JoyLTX_Keyframes": "JoyLTX Keyframes (end / mid frames)",
                              "JoyLTX_ImageFolder": "JoyLTX Image Folder (one image per shot)"}
