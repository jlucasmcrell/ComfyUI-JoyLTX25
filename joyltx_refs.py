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


def _load(path, w, h):
    im = Image.open(path); im = ImageOps.exif_transpose(im).convert("RGB")
    im = ImageOps.fit(im, (w, h), Image.LANCZOS, centering=(0.5, 0.35))   # faces sit high: bias the crop up
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


class JoyLTX_RefsByName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompts": ("STRING", {"forceInput": True, "tooltip": "The writer's shot prompts (JSON or --- list)."}),
            "refs_root": ("STRING", {"default": "joyecho_refs", "tooltip":
                          "Folder under ComfyUI/input holding one sub-folder per character (the folder NAME is the "
                          "name the node looks for in each shot prompt, case-insensitive)."}),
            "pick": (["one per character (by seed)", "first", "random per shot", "random by seed"], {"default": "one per character (by seed)", "tooltip":
                     "one per character = the SAME photo for a character in every shot (best identity)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 32, "tooltip": "Render size (pass 1); the photos are fit to it."}),
            "height": ("INT", {"default": 544, "min": 64, "max": 4096, "step": 32}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ref_images", "ref_mask", "report", "prompts")
    FUNCTION = "run"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "One reference photo per shot, picked by the character names in the prompts from input/<refs_root>/<name>/."

    def run(self, prompts, refs_root="joyecho_refs", pick="random by seed", seed=0, width=960, height=544):
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
                text = text[:m.start()] + text[m.end():]
                text = re.sub(r"^[\s,.;:-]+", "", text)
            clean.append(text)
            low = text.lower()
            if hit is None:
                # otherwise: the most-mentioned character; ties -> the earliest mention
                counts = {}
                for n in names:
                    ms = [mm.start() for mm in re.finditer(r"(?<![a-z0-9_])" + re.escape(n) + r"(?![a-z0-9_])", low)]
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
            imgs.append(_load(f, width, height)); mask.append(hit)
            rep.append("shot %d: %s <- %s" % (i + 1, hit, os.path.basename(f)))
        if not imgs:
            imgs = [torch.zeros([1, height, width, 3])]; mask = ["-"]; rep = ["no shots"]
        report = "refs_root=%s  characters=%s\n%s" % (root, ", ".join(sorted(chars)) or "(none)", "\n".join(rep))
        print("[JoyLTX RefsByName] " + report.replace("\n", " | "), flush=True)
        out_prompts = json.dumps({"prompts": clean}) if clean else (prompts or "")
        return (torch.cat(imgs, 0), ",".join(mask), report, out_prompts)


NODE_CLASS_MAPPINGS = {"JoyLTX_RefsByName": JoyLTX_RefsByName}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_RefsByName": "JoyLTX Refs by Name (input/<root>/<character>/)"}
