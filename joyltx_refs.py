"""JoyLTX_RefsByName: automatic reference photos by character name.

Reads the shot prompts, finds character names that have a folder under <ComfyUI>/input/<refs_root>/<name>/
(case-insensitive), and returns ONE reference image per shot (the first named character that has a folder;
`pick` = first / random per shot / fixed by seed) plus a mask string of character names per shot ("alice,bob,-") the sampler uses for its per-character identity and voice locks.
Write "[ref: alice]" (or "@alice") in a shot to force its photo; the marker is stripped from the `prompts` output.
Feed `ref_images` + `ref_mask` into the JoyLTX Multishot sampler: each shot's photo becomes an in-context
keyframe at frame 0 (LTX-2.5 AddGuide, appended tokens, cropped after pass 1) at `ref_strength`, so the
person in the folder is the person in the shot - no LoRA, no captions needed.
"""
import os, re, random, json
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_SPLIT = re.compile(r"^\s*---+\s*$", re.M)


def _parse_prompts(text):
    t = (text or "").strip()
    if not t:
        return []
    try:
        data = json.loads(t)
        if isinstance(data, dict):
            data = data.get("prompts") or data.get("shots") or []
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return [p.strip() for p in _SPLIT.split(t) if p.strip()]


def _roots():
    """Folders under ComfyUI/input that hold at least one sub-folder of images - the character libraries."""
    base = folder_paths.get_input_directory(); out = []
    try:
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if not os.path.isdir(p) or d.startswith((".", "_")):
                continue
            for sub in os.listdir(p):
                sp = os.path.join(p, sub)
                if os.path.isdir(sp) and any(f.lower().endswith(_EXTS) for f in os.listdir(sp)):
                    out.append(d); break
    except Exception:
        pass
    return out or ["joyecho_refs"]


def _characters(root):
    out = {}
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if os.path.isdir(p):
                imgs = sorted(f for f in os.listdir(p) if f.lower().endswith(_EXTS))
                if imgs:
                    out[d.lower()] = [os.path.join(p, f) for f in imgs]
    return out


def _load(path, w, h, crop="portrait"):
    im = Image.open(path); im = ImageOps.exif_transpose(im).convert("RGB")
    if crop == "portrait":
        # keep the person, drop the set: the central upper 60 % of the photo (faces sit high), then fit
        W, H = im.size
        cw, ch = int(W * 0.6), int(H * 0.6)
        left = (W - cw) // 2; top = int((H - ch) * 0.25)
        im = im.crop((left, top, left + cw, top + ch))
    im = ImageOps.fit(im, (w, h), Image.LANCZOS, centering=(0.5, 0.35))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


class JoyLTX_RefsByName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "The writer's shot prompts (JSON or --- list). Normally wired from the writer or Prompts from File."}),
            "refs_root": (_roots(), {"tooltip":
                          "Folder under ComfyUI/input holding one sub-folder per character (the folder NAME is the "
                          "name the node looks for in each shot prompt, case-insensitive). Restart or refresh ComfyUI "
                          "to see folders you just added."}),
            "pick": (["one per character (by seed)", "first", "random per shot", "random by seed"], {"default": "one per character (by seed)", "tooltip":
                     "one per character = the SAME photo for a character in every shot (best identity)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 32, "tooltip": "Render size (pass 1); the photos are fit to it."}),
            "height": ("INT", {"default": 544, "min": 64, "max": 4096, "step": 32}),
        }, "optional": {
            "crop": (["portrait (keep the person, drop the set)", "full photo"], {"default": "portrait (keep the person, drop the set)", "tooltip":
                     "portrait = the central upper part of each photo, so the reference carries the face and not the room "
                     "(a full scene photo drags its set into the shot)."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ref_images", "ref_mask", "report", "prompts")
    FUNCTION = "run"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "One reference photo per shot, picked by the character names in the prompts from input/<refs_root>/<name>/."

    PICKS = ["one per character (by seed)", "first", "random per shot", "random by seed"]

    def run(self, prompts, refs_root="joyecho_refs", pick="random by seed", seed=0, width=960, height=544, crop="portrait (keep the person, drop the set)"):
        if pick not in self.PICKS or not isinstance(refs_root, str) or refs_root in self.PICKS:
            raise RuntimeError("[JoyLTX RefsByName] this node's widget values are shifted (refs_root=%r, pick=%r). "
                               "Re-pick them in the node and save the workflow." % (refs_root, pick))
        cmode = "portrait" if str(crop).startswith("portrait") else "full"
        root = os.path.join(folder_paths.get_input_directory(), refs_root or "joyecho_refs")
        chars = _characters(root)
        shots = _parse_prompts(prompts)
        rng = random.Random(seed)
        imgs, mask, rep = [], [], []
        names = sorted(chars.keys(), key=len, reverse=True)
        fixed = {n: random.Random(seed * 7919 + k).randrange(len(chars[n])) for k, n in enumerate(sorted(chars))}
        clean = []
        for i, text in enumerate(shots):
            # explicit pick: "[ref: name]" or "@name" anywhere in the shot wins (and is stripped from the prompt)
            hit = None
            m = re.search(r"\[\s*ref\s*:\s*([A-Za-z0-9_\-]+)\s*\]|(?<![A-Za-z0-9_])@([A-Za-z0-9_\-]+)", text)
            if m:
                cand = (m.group(1) or m.group(2)).lower()
                if cand in chars:
                    hit = cand
                else:                                   # "[ref: zara_rift]" -> the zara folder
                    base = cand
                    while "_" in base and base not in chars:
                        base = base.rsplit("_", 1)[0]
                    if base in chars:
                        hit = base
                text = text[:m.start()] + text[m.end():]
                text = re.sub(r"^[\s,.;:-]+", "", text)
            # safety net: a "looks exactly as in the reference photographs" sentence whose subject
            # has NO folder points the render model at photographs that are not attached - the
            # sentence carries no information and costs the character their real description
            # upstream. Strip it and say so, so the folder (or the premise) gets fixed.
            for pm in list(re.finditer(
                    r"([A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*)?)"
                    r"((?:,[^.,]{0,60}){0,2},?\s+looks? exactly as in the reference photographs?[^.]*\.\s*)",
                    text))[::-1]:
                toks = []
                for t in pm.group(1).split():
                    t = t.lower().strip(",.")
                    toks.append(t)
                    while "_" in t:                     # zara_rift -> zara
                        t = t.rsplit("_", 1)[0]
                        toks.append(t)
                if not any(t in chars for t in toks):
                    text = text[:pm.start()] + text[pm.end():]
                    rep.append("shot %d: removed the photo pointer for '%s' - no folder %s"
                               % (i + 1, pm.group(1), os.path.join(root, toks[-1])))
            clean.append(text)
            low = text.lower()
            if hit is None:
                # otherwise: the most-mentioned character; ties -> the earliest mention
                counts = {}
                for n in names:
                    # trailing _suffix tolerated: the older corpus writes LoRA triggers
                    # ("zara_rift", "alana_rift"), which matched no folder and cast nobody.
                    ms = [mm.start() for mm in re.finditer(
                        r"(?<![a-z0-9_])" + re.escape(n) + r"(?:_[a-z0-9]+)*(?![a-z0-9])", low)]
                    if ms:
                        counts[n] = (len(ms), -ms[0])
                if counts:
                    hit = max(counts, key=lambda k: counts[k])
            if hit is None:
                imgs.append(torch.zeros([1, height, width, 3])); mask.append("-"); rep.append("shot %d: (no named character with a folder)" % (i + 1))
                continue
            files = chars[hit]
            if pick.startswith("one per character"):
                f = files[fixed[hit]]
            elif pick == "first":
                f = files[0]
            elif pick == "random per shot":
                f = random.choice(files)
            else:
                f = files[rng.randrange(len(files))]
            imgs.append(_load(f, width, height, cmode)); mask.append(hit)
            rep.append("shot %d: %s <- %s" % (i + 1, hit, os.path.basename(f)))
        if not imgs:
            imgs = [torch.zeros([1, height, width, 3])]; mask = ["-"]; rep = ["no shots"]
        report = "refs_root=%s  characters=%s\n%s" % (root, ", ".join(sorted(chars)) or "(none)", "\n".join(rep))
        print("[JoyLTX RefsByName] " + report.replace("\n", " | "), flush=True)
        out_prompts = json.dumps({"prompts": clean}) if clean else (prompts or "")
        return (torch.cat(imgs, 0), ",".join(mask), report, out_prompts)


NODE_CLASS_MAPPINGS = {"JoyLTX_RefsByName": JoyLTX_RefsByName}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_RefsByName": "JoyLTX Refs by Name"}

class JoyLTX_RefImage:
    """Manual reference: one loaded image (or a batch) instead of the folder library.

    Wire a Load Image here and the outputs to the sampler's ref_images / ref_mask - same crop and
    sizing as Refs by Name, no folders involved. One image = every shot; a batch = one per shot in
    order (the last repeats). Give the character's name and the sampler's per-character identity
    and voice locks apply to them; leave it blank and the photo is applied without the locks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The reference photo (Load Image). A batch = one per shot."}),
            "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 32,
                      "tooltip": "Render size (pass 1) - wire from MASTER CONTROLS like Refs by Name."}),
            "height": ("INT", {"default": 544, "min": 64, "max": 4096, "step": 32}),
        }, "optional": {
            "character": ("STRING", {"default": "", "tooltip":
                          "Optional: who this is. Named, the sampler keeps their face AND voice locked "
                          "across shots (same as a folder character). Blank: photo only, no locks."}),
            "crop": (["portrait (keep the person, drop the set)", "full photo"],
                     {"default": "portrait (keep the person, drop the set)"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("ref_images", "ref_mask", "report")
    FUNCTION = "run"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "One reference photo by hand - no folder library, same crop/sizing as Refs by Name."

    def run(self, image, width, height, character="", crop="portrait (keep the person, drop the set)"):
        t = image[..., :3].float()
        if str(crop).startswith("portrait"):
            B, H, W, C = t.shape
            ch, cw = int(H * 0.6), int(W * 0.6)
            top = int((H - ch) * 0.25); left = (W - cw) // 2
            t = t[:, top:top + ch, left:left + cw, :]
        # cover-fit to the render size, face-high centering (matches Refs by Name's ImageOps.fit)
        B, H, W, C = t.shape
        sc = max(width / W, height / H)
        nw, nh = max(width, int(round(W * sc))), max(height, int(round(H * sc)))
        t = torch.nn.functional.interpolate(t.movedim(-1, 1), size=(nh, nw), mode="bilinear",
                                            align_corners=False).movedim(1, -1)
        left = max(0, (nw - width) // 2); top = max(0, int((nh - height) * 0.35))
        t = t[:, top:top + height, left:left + width, :].contiguous()
        name = re.sub(r"[^a-z0-9_-]", "", str(character or "").strip().lower())
        mask = name if name else "1"
        report = "manual ref: %d image(s) at %dx%d as %s" % (t.shape[0], width, height,
                                                             ("'%s' (identity+voice locked)" % name) if name else "unnamed (no locks)")
        print("[JoyLTX RefImage] " + report, flush=True)
        return (t, mask, report)


NODE_CLASS_MAPPINGS["JoyLTX_RefImage"] = JoyLTX_RefImage
NODE_DISPLAY_NAME_MAPPINGS["JoyLTX_RefImage"] = "JoyLTX Ref Image (manual)"

