"""JoyEcho Plate Picker - auto-inject a verified surface plate as reference_image.

THE PROBLEM IT SOLVES
    Operator, 2026-07-28: "if we have a library of samples, can we not use them
    as auto-injected references based on the prompts?"  And earlier, the sharper
    version: "are we going to remember to do that every single time someone is
    standing on a lawn?"

    Measured 2026-08-01: feeding a Z-Image surface plate in as reference_image
    beats both the stock baseline and the retrained surfaces LoRA on vegetation,
    and it replicates - 4 seeds out of 4. Baseline renders a dry wiry mat; the
    plate renders grass with individual blades and real shadow depth.

    The blocker was never the mechanism, it was SELECTION. Picking plates by
    grepping captions for a material keyword failed badly: three of four picks
    did not show their claimed material at all, because the captions came from
    the GENERATION PROMPT rather than the image, and Z-Image renders concrete or
    asphalt about a third of the time regardless of what was asked for.

WHY THIS USES THE RE-CAPTIONED SET
    surfaces_v2b_recap captions are written by a vision model looking at the
    IMAGE. They say what is actually there. A plate labelled "leaf litter" in
    that set really is leaf litter, which is exactly the property a lookup table
    needs and the prompt-derived captions did not have.

MATCHING
    Deliberately dumb and deterministic: token overlap between the prompt's
    ground-surface phrases and each plate's material clause, with a small bonus
    for exact material-word hits. No embeddings - a lookup you cannot predict is
    worse than one that occasionally misses, because you cannot debug it. The
    chosen plate is printed every run.

    Returns a passthrough NO-OP (zero image + empty name) when nothing scores
    above `min_score`, so an unmatched prompt renders exactly as it would have
    without this node. Silent wrong plates are worse than no plate.
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch

# Ground/surface words worth matching on. Deliberately narrow: matching on
# generic scene nouns pulls in plates that merely mention a place.
_SURFACE_HINTS = re.compile(
    r"\b(asphalt|tarmac|blacktop|concrete|pavement|paving|cobble|brick|gravel|"
    r"shingle|dirt|mud|earth|soil|clay|sand|dune|grass|lawn|turf|meadow|moss|"
    r"leaf litter|leaves|forest floor|woodland|pine needles|snow|ice|slush|"
    r"lino|linoleum|carpet|tile|tiles|floorboards|decking|plywood|steel|metal|"
    r"grating|rubber|gravelly|scree|rock|stone|slate|straw|hay|stubble|"
    r"puddle|water|wet)\b", re.I)

_STOP = {"the", "a", "an", "and", "with", "of", "in", "on", "its", "into",
         "over", "under", "across", "is", "are", "at", "to", "by", "from"}

# Condition words that CHANGE THE SEASON OR WEATHER of a shot. Matching material
# alone is not enough: the first dry run picked "frozen muddy water" for a wet
# mud track and "frost-covered meadow grass" for dry wind-flattened grass. Right
# material, wrong world - and a reference_image drags its whole look across, so
# a frost plate would put winter into a summer scene. If the plate asserts one
# of these and the prompt does not, penalise it hard.
_CONDITION_CONFLICT = re.compile(
    r"\b(frost|frosted|frozen|snow|snowy|snow-covered|ice|icy|slush|"
    r"sunlit|sun-bleached|scorched|charred|burnt|flooded|submerged)\b", re.I)


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower())
            if len(w) > 2 and w not in _STOP}


class JoyEcho_PlatePicker:
    """Pick a verified surface plate matching the prompt's ground material."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip":
                           "The scene prompt. Only its ground/surface phrases "
                           "are read; everything else is ignored."}),
                "plate_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Folder of <id>.png + <id>.txt pairs. Use the "
                               "RE-CAPTIONED set: those captions describe the "
                               "image, not the prompt that failed to produce it."}),
                "min_score": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0,
                              "step": 0.5, "tooltip":
                              "Below this the node returns nothing and the "
                              "render proceeds unchanged. A wrong plate is "
                              "worse than no plate."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("plate", "plate_name", "matched_material")
    FUNCTION = "pick"
    CATEGORY = "JoyAI-Echo/RiftCast"

    _cache: dict = {}

    # Plates containing people, limbs or cast human shadows. A reference_image
    # drags its content across, so injecting one of these puts a figure into a
    # scene that asked for an empty one - the exact failure the v2b regeneration
    # was supposed to fix. Found by eye: four pixel heuristics all failed at it.
    # The picker read this list only AFTER it selected surface_02910 ("person
    # kneeling on wet sand") for a mud prompt during its first dry run.
    # File format: one plate id per line, '#' comments allowed.
    DENYLIST_FILE = "plate_denylist.txt"

    @staticmethod
    def _norm_id(s: str) -> str:
        """Plate ids appear as surface_02910, surface_02910_ and surface_02910_.png
        depending on whether they came from a filename or from prose. Normalise
        or the denylist silently matches nothing - which it did on first run,
        and the contaminated plate was selected anyway."""
        s = os.path.basename(s.strip())
        for ext in (".png", ".txt"):
            if s.lower().endswith(ext):
                s = s[: -len(ext)]
        return s.rstrip("_").lower()

    @classmethod
    def _denylist(cls, plate_dir: str) -> set:
        out = set()
        for cand in (os.path.join(plate_dir, cls.DENYLIST_FILE),
                     os.path.join(os.path.dirname(__file__), cls.DENYLIST_FILE)):
            if os.path.exists(cand):
                for l in open(cand, encoding="utf-8", errors="replace"):
                    l = l.split("#")[0].strip()
                    if l:
                        out.add(cls._norm_id(l))
        return out

    @classmethod
    def _index(cls, plate_dir: str):
        key = (plate_dir, os.path.getmtime(plate_dir) if os.path.isdir(plate_dir) else 0)
        if key in cls._cache:
            return cls._cache[key]
        idx = []
        deny = cls._denylist(plate_dir)
        skipped = 0
        if os.path.isdir(plate_dir):
            for fn in sorted(os.listdir(plate_dir)):
                if not fn.lower().endswith(".txt"):
                    continue
                if cls._norm_id(fn) in deny:
                    skipped += 1
                    continue
                png = os.path.join(plate_dir, fn[:-4] + ".png")
                if not os.path.exists(png):
                    continue
                try:
                    cap = open(os.path.join(plate_dir, fn), encoding="utf-8",
                               errors="replace").read().strip()
                except Exception:
                    continue
                material = cap.split(",")[0].strip()
                idx.append((png, material, _tokens(material)))
        cls._cache.clear()
        cls._cache[key] = idx
        return idx

    def pick(self, prompt, plate_dir, min_score, enabled=True):
        empty = (torch.zeros((1, 8, 8, 3), dtype=torch.float32), "", "")
        if not enabled:
            return empty
        idx = self._index(plate_dir)
        if not idx:
            print(f"[PlatePicker] no <id>.png/<id>.txt pairs in {plate_dir} "
                  f"- passing through.", flush=True)
            return empty

        hints = {h.lower() for h in _SURFACE_HINTS.findall(prompt or "")}
        if not hints:
            print("[PlatePicker] prompt names no ground surface - passing through.",
                  flush=True)
            return empty
        want = _tokens(" ".join(hints))

        best, best_score, best_mat = None, 0.0, ""
        for png, material, toks in idx:
            overlap = len(want & toks)
            if not overlap:
                continue
            # exact surface-word hits count double; they are the signal, the
            # rest of the caption is condition/colour wording
            score = overlap + sum(1.0 for h in hints if h in material.lower())
            # season/weather the prompt never asked for: a reference_image drags
            # its whole look across, so this is a mismatch, not a nuance
            for cw in _CONDITION_CONFLICT.findall(material):
                if cw.lower() not in (prompt or "").lower():
                    score -= 2.0
            if score > best_score:
                best, best_score, best_mat = png, score, material

        if best is None or best_score < min_score:
            print(f"[PlatePicker] best score {best_score:.1f} < {min_score} "
                  f"for {sorted(hints)} - passing through (no plate).", flush=True)
            return empty

        try:
            from PIL import Image
            im = Image.open(best).convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr)[None, ...]
        except Exception as e:
            print(f"[PlatePicker] could not load {best} ({e}) - passing through.",
                  flush=True)
            return empty

        name = os.path.basename(best)
        print(f"[PlatePicker] {sorted(hints)} -> {name}  "
              f"(score {best_score:.1f}) material={best_mat!r}", flush=True)
        return (t, name, best_mat)


NODE_CLASS_MAPPINGS = {"JoyEcho_PlatePicker": JoyEcho_PlatePicker}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyEcho_PlatePicker": "JoyEcho Plate Picker (auto surface reference)"}
