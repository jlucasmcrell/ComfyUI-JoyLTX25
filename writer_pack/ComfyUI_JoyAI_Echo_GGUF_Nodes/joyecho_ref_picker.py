"""JoyEcho Reference Picker - auto-select a character reference image.

Feeds JoyEcho_Generate's reference_image input in LPFF batch queues:
  LPFF block carries `name: alice` -> UnzipPrompt name output -> this node ->
  picks an image from <ComfyUI>/input/joyecho_refs/alice/ -> IMAGE out.

Resolution order per run:
  1. `character` input (usually UnzipPrompt's name output), lowercased.
     LPFF quirk: blocks WITHOUT a name: line emit the prompt FILENAME here -
     that never matches a folder, so it falls through cleanly.
  2. scan `prompt_text` for any refs-folder name as a whole word (longest first).
  3. `fallback_image` input if wired.
  4. clear error.

Pick strategies: by_seed (sorted files, seed % count - reproducible, vary the
seed to vary the ref), first, newest.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

_REFS_SUBDIR = "joyecho_refs"
_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _refs_root(custom_root: str = "") -> Path:
    if custom_root and custom_root.strip():
        return Path(custom_root.strip())
    d = Path(folder_paths.get_input_directory()) / _REFS_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _character_dirs(root: Path) -> list[str]:
    try:
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    except OSError:
        return []


def _images_in(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() in _EXTS)
    except OSError:
        return []


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1, H, W, C]


_PICK_AUTO = "(from character input / prompt scan)"
_PICK_NONE = "(none / disabled)"
_DEFAULT_REFS_ROOT = ""  # empty = <ComfyUI>/input/joyecho_refs/


def _character_options() -> list[str]:
    """Dropdown options: the character folders in the refs root (default
    <ComfyUI>/input/joyecho_refs/, one subfolder per character), lowercased
    (matching is case-insensitive on Windows). Built at class definition;
    press R in ComfyUI after adding a new character folder."""
    root = Path(_DEFAULT_REFS_ROOT) if _DEFAULT_REFS_ROOT else _refs_root("")
    if not root.is_dir():
        root = _refs_root("")
    opts = {d.lower() for d in _character_dirs(root)}
    return [_PICK_AUTO, _PICK_NONE] + sorted(opts)


class JoyEcho_RefPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pick": (["by_seed", "first", "newest"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "Used by by_seed: index = seed % image count."}),
            },
            "optional": {
                "refs_root": ("STRING", {
                    "default": _DEFAULT_REFS_ROOT,
                    "tooltip": "Root folder holding one subfolder per character. Empty = ComfyUI/input/joyecho_refs/. Folder-name matching is case-insensitive on Windows.",
                }),
                "on_no_match": (["no_reference", "error"], {
                    "default": "no_reference",
                    "tooltip": "When no character matches and no fallback_image is wired: "
                               "no_reference = output nothing (Generate simply skips identity "
                               "seeding for this item; the batch keeps running). error = stop the run.",
                }),
                "character": ("STRING", {"default": "",
                                         "tooltip": "Character folder name (e.g. alice). TYPE it here for a "
                                                    "manual pick, or right-click the node > 'Convert character "
                                                    "to input' and wire PromptSource's character output for "
                                                    "automatic per-item picks. Overridden by character_pick "
                                                    "when that dropdown is not on its auto setting."}),
                "prompt_text": ("STRING", {"default": "", "forceInput": True,
                                           "tooltip": "Fallback: scanned for any refs folder name as a whole word. "
                                                      "Only used when NO character is explicitly selected."}),
                "fallback_image": ("IMAGE",),
                "character_pick": (_character_options(), {
                    "default": _PICK_AUTO,
                    "tooltip": "Pick the character folder from a dropdown - survives model "
                               "refreshes (a typed name can get wiped; a blank picker then "
                               "silently falls back to the prompt scan and can grab the wrong "
                               "character). Leave on the auto setting to use the character "
                               "input / prompt scan instead.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("reference_image", "picked_path", "ref_shots",)
    FUNCTION = "pick_ref"
    CATEGORY = "JoyAI-Echo"

    @classmethod
    def IS_CHANGED(cls, pick, seed, on_no_match="no_reference", refs_root="", character="", prompt_text="", fallback_image=None, character_pick=_PICK_AUTO):
        # Re-run when anything that affects the pick changes. prompt_text MUST
        # be in the signature: without it, ComfyUI can serve a CACHED pick from
        # a previous queue item (character A's ref surfacing in character B's render).
        root = _refs_root(refs_root)
        # Wired inputs (e.g. character from PromptSource) resolve to None while
        # ComfyUI evaluates IS_CHANGED before upstream nodes have run - guard
        # everything, or the exception surfaces as a per-item WARNING and the
        # node loses caching entirely.
        sig = [pick, str(seed), str(root), (character or "").strip().lower(),
               str(character_pick), str(hash(prompt_text or ""))]
        for d in _character_dirs(root):
            folder = root / d
            imgs = _images_in(folder)
            sig.append(f"{d}:{len(imgs)}:{max((p.stat().st_mtime for p in imgs), default=0)}")
        return "|".join(sig)

    def pick_ref(self, pick, seed, on_no_match="no_reference", refs_root="", character="", prompt_text="", fallback_image=None, character_pick=_PICK_AUTO):
        root = _refs_root(refs_root)
        dirs = _character_dirs(root)

        # Deliberate off-switch: lets a spare picker (e.g. the second one from
        # a two-character wiring) be disabled without rewiring or clearing
        # typed fields. Never errors, never scans, never uses fallback_image.
        if character_pick == _PICK_NONE:
            print("[JoyEcho] RefPicker: disabled via character_pick; emitting no reference.",
                  flush=True)
            return (None, "(disabled)", "")

        chosen_dir = None
        explicit = False
        # 1) Dropdown wins: a combo value persists across model refreshes,
        #    where a typed STRING has been observed to get wiped.
        if character_pick and character_pick != _PICK_AUTO:
            want = character_pick.strip().lower()
            explicit = True
        else:
            want = (character or "").strip().lower()
            # LPFF quirk: blocks without a `name:` line emit the prompt FILENAME
            # as the name - anything path/file-shaped is not a character.
            if any(s in want for s in ("\\", "/", ".txt", ".json")):
                want = ""
            explicit = bool(want)
        if want and (root / want).is_dir():
            chosen_dir = root / want

        # An EXPLICITLY named character that doesn't match a folder must NOT
        # silently fall through to the prompt scan - that's how a wiped/typo'd
        # name turns into the wrong character's face in a render. Treat it as
        # no-match instead (loud, and on_no_match governs).
        if chosen_dir is None and explicit:
            print(f"[JoyEcho] RefPicker: explicit character {want!r} has no folder under "
                  f"{root} - NOT falling back to prompt scan.", flush=True)
            if fallback_image is not None:
                return (fallback_image, "(fallback_image)", "0")
            if on_no_match == "no_reference":
                return (None, "(no reference)", "")
            raise ValueError(
                f"RefPicker: explicit character {want!r} matched no folder in {root}. "
                f"Available: {dirs or '(none)'}")

        if chosen_dir is None and prompt_text:
            # Ignore names spoken INSIDE dialogue: absent characters get
            # mentioned in quotes ("Rae thinks I am imagining it"), while
            # on-screen characters are named in the descriptive prose. Strip
            # JSON-escaped quotes, plain double quotes, and says,-introduced
            # single-quoted lines before scanning.
            scrub = prompt_text
            # JSON script payloads ({"prompts": [...]}) must be PARSED before
            # scanning: on the raw JSON string, the plain-quote stripper below
            # matches the JSON's own quoted strings - i.e. the entire prompt
            # body - and deletes everything, so no character ever matches.
            try:
                import json as _json
                _d = _json.loads(prompt_text)
                if isinstance(_d, dict):
                    _arr = _d.get("prompts") or _d.get("shots")
                    if isinstance(_arr, list) and _arr:
                        scrub = " ".join(str(x) for x in _arr)
            except (ValueError, TypeError):
                pass
            scrub = re.sub(r"<d>.*?</d>", " ", scrub, flags=re.S)      # H3 dialogue tags
            scrub = re.sub(r'\\"(?:[^"\\]|\\.)*?\\"', " ", scrub)      # \"...\" (JSON-escaped)
            scrub = re.sub(r'"(?:[^"\\]|\\.)*?"', " ", scrub)          # "..."
            scrub = re.sub(r"says,\s*'(?:[^'])*?'", " ", scrub)        # says, '...'
            low = scrub.lower()
            # EVERY character named in the descriptive prose gets ONE reference
            # (dialogue already stripped above, so absent characters spoken
            # about in quotes never match). Solo scene -> 1 ref, two-character
            # scene -> 2 refs, no names -> none. Ordered by first mention so
            # the protagonist's ref leads. One picker, zero mode-flipping.
            # Per-shot prose (JSON payloads) so each character's FIRST SHOT is
            # known: refs are then injected where the character ENTERS the video
            # instead of all at shot 1 (a wrong-scene ref at shot 1 is the
            # wrong-character-opening failure). Plain-text briefs -> shot 0.
            _shot_texts = None
            _refs_map = {}
            try:
                import json as _json2
                _d2 = _json2.loads(prompt_text)
                _arr2 = (_d2.get("prompts") or _d2.get("shots")) if isinstance(_d2, dict) else None
                if isinstance(_arr2, list) and _arr2:
                    _shot_texts = [re.sub(r'"[^"]*"', " ", str(x)).lower() for x in _arr2]
                # Optional script-carried ref pinning: {"refs": {"alice": "alice_wide_05.png"}}
                # A full-scene reference SETS the render's scene, so scripted
                # episodes pin a scene-matched file instead of rolling by_seed.
                if isinstance(_d2, dict) and isinstance(_d2.get("refs"), dict):
                    _refs_map = {str(k).strip().lower(): str(v).strip()
                                 for k, v in _d2["refs"].items()}
            except (ValueError, TypeError):
                pass
            matched = []  # (first_pos, dirname, [injection shots])
            for d in dirs:
                pat = r"\b" + re.escape(d.lower()) + r"\b"
                hits = [m.start() for m in re.finditer(pat, low)]
                if hits:
                    inject = [0]
                    if _shot_texts:
                        # All appearance shots -> inject at the FIRST, plus at any
                        # RE-ENTRY after >=3 absent shots (the memory bank's rolling
                        # window is 4; official example scripts never exceed a
                        # 2-shot absence, so 3+ means the bank has likely lost them).
                        appear = [si for si, st in enumerate(_shot_texts) if re.search(pat, st)]
                        if appear:
                            inject = [appear[0]]
                            for prev, cur in zip(appear, appear[1:]):
                                if cur - prev - 1 >= 3:
                                    inject.append(cur)
                    matched.append((hits[0], d, inject))
            # Force-inject scripted refs whose folder was NOT named in the prose.
            # Entity/environment scenes describe "a black horse" or "a bandstand",
            # never the folder name, so the prose scan above misses them - but the
            # scene still carries a JSON refs block ({"kelpie": "..."}) pinning the
            # exact still. Honor those directly so the reference loads without
            # contorting the prose. Prose-matched characters keep their real
            # positions and still lead; forced refs sort last and inject at shot 1.
            if _refs_map:
                _lower_to_dir = {d.lower(): d for d in dirs}
                _already = {d.lower() for _, d, _ in matched}
                for _name in _refs_map:
                    if _name in _already:
                        continue
                    _real = _lower_to_dir.get(_name)
                    if _real is not None:
                        matched.append((10**9, _real, [0]))
                    else:
                        print(f"[JoyEcho] RefPicker: scripted ref {_name!r} has no folder "
                              f"under {root}; skipping.", flush=True)
            matched.sort()
            if matched:
                picked = []   # (dirname, path) per injection entry
                sched = []
                for _, d, inject in matched[:4]:  # at most 4 characters
                    folder = root / d
                    imgs = _images_in(folder)
                    if not imgs:
                        print(f"[JoyEcho] RefPicker: {d}/ matched but is empty; skipping.", flush=True)
                        continue
                    path = None
                    pin = _refs_map.get(d.lower())
                    if pin:
                        cand = folder / pin
                        if cand.is_file():
                            path = cand
                        else:
                            print(f"[JoyEcho] RefPicker: pinned ref {d}/{pin} not found; "
                                  f"falling back to {pick}.", flush=True)
                    if path is None:
                        if pick == "first":
                            path = imgs[0]
                        elif pick == "newest":
                            path = max(imgs, key=lambda p: p.stat().st_mtime)
                        else:  # by_seed
                            path = imgs[seed % len(imgs)]
                    for shot in inject:
                        if len(picked) >= 6:  # Generate accepts at most 6 scheduled entries
                            break
                        picked.append((d, path))
                        sched.append(shot)
                if picked:
                    pils = []
                    for d, path in picked:
                        img = Image.open(path)
                        pils.append(ImageOps.exif_transpose(img).convert("RGB"))
                    # An IMAGE batch tensor must be uniform: resize any
                    # mismatched refs to the first ref's dimensions.
                    w0, h0 = pils[0].size
                    pils = [p if p.size == (w0, h0) else p.resize((w0, h0), Image.LANCZOS)
                            for p in pils]
                    tensors = [torch.from_numpy(np.asarray(p).astype(np.float32) / 255.0)[None, ...]
                               for p in pils]
                    names = ", ".join(f"{d} -> {path.name} @shot{s+1}" for (d, path), s in zip(picked, sched))
                    print(f"[JoyEcho] RefPicker: prompt scan matched {len(picked)} character(s): "
                          f"{names} ({pick}).", flush=True)
                    return (torch.cat(tensors, dim=0), "; ".join(str(p) for _, p in picked),
                            ",".join(str(s) for s in sched))

        if chosen_dir is None:
            if fallback_image is not None:
                print("[JoyEcho] RefPicker: no character match; using fallback_image.", flush=True)
                return (fallback_image, "(fallback_image)", "0")
            if on_no_match == "no_reference":
                print(f"[JoyEcho] RefPicker: no character match (character={character!r}); "
                      f"continuing WITHOUT a reference.", flush=True)
                return (None, "(no reference)", "")
            raise ValueError(
                f"RefPicker: no reference folder matched. character={character!r}, "
                f"available folders in {root}: {dirs or '(none - create input/joyecho_refs/<name>/)'}"
            )

        imgs = _images_in(chosen_dir)
        if not imgs:
            if fallback_image is not None:
                print(f"[JoyEcho] RefPicker: {chosen_dir.name}/ is empty; using fallback_image.", flush=True)
                return (fallback_image, "(fallback_image)", "0")
            if on_no_match == "no_reference":
                print(f"[JoyEcho] RefPicker: {chosen_dir.name}/ is empty; continuing WITHOUT a reference.", flush=True)
                return (None, "(no reference)", "")
            raise ValueError(f"RefPicker: no images in {chosen_dir} (put .png/.jpg refs there).")

        if pick == "first":
            path = imgs[0]
        elif pick == "newest":
            path = max(imgs, key=lambda p: p.stat().st_mtime)
        else:  # by_seed
            path = imgs[seed % len(imgs)]

        print(f"[JoyEcho] RefPicker: {chosen_dir.name} -> {path.name} "
              f"({pick}, {len(imgs)} candidates).", flush=True)
        return (_load_image(path), str(path), "0")


NODE_CLASS_MAPPINGS = {"JoyEcho_RefPicker": JoyEcho_RefPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_RefPicker": "JoyEcho Reference Picker (auto by character)"}
