"""JoyLTX_StorySource - one panel that decides where a run's words come from.

Three sources, one switch:
  scene idea (the box)  - type a premise; the writer turns it into shot prompts.
  idea file             - a file of story IDEAS (--- between them, or JSON); `index` picks one, the writer
                          turns that idea into shot prompts. Point `index` at a Primitive on increment and
                          queue once per idea to render a whole file unattended.
  shot file             - a file of FINISHED shot prompts; the writer is skipped entirely.

Outputs feed, in order: story_idea -> the writer, use_file -> the prompts switch (off = writer, on = file),
file_prompts -> that switch's ON side, refs_attached -> the writer (names the characters and points at the
reference photographs instead of inventing faces), name -> anything that wants the source's file name.

Either path may name a folder instead of a file: then `index` picks the file (sorted, wraps).
"""
import json
import os
import re

import folder_paths

_SPLIT = re.compile(r"^\s*---+\s*$", re.M)
_TEXT = (".txt", ".json")


def _catalog():
    """Every .txt / .json under ComfyUI/input (depth 3), plus each folder that holds some, as `folder/`.
    Folder entries let INDEX walk a whole folder. Refresh ComfyUI to pick up new files."""
    base = folder_paths.get_input_directory()
    files, dirs = [], set()
    for root, dn, fn in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel.count(os.sep) > 2:
            dn[:] = []
            continue
        dn[:] = [d for d in dn if not d.startswith((".", "_"))]
        hits = [f for f in sorted(fn) if f.lower().endswith(_TEXT)]
        if hits:
            r = "" if rel == "." else rel.replace(os.sep, "/") + "/"
            if r:
                dirs.add(r)
            files += [r + f for f in hits]
    return sorted(dirs) + sorted(files)


def _entries():
    c = _catalog()
    return c if c else ["(no .txt / .json in ComfyUI/input)"]
SOURCES = ["scene idea (the box)", "idea file (the writer writes each idea)", "shot file (skip the writer)"]


def _resolve(path):
    p = str(path or "").strip().strip('"')
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.join(folder_paths.get_input_directory(), p)


def _pick_file(path, index):
    """A file stays itself; a folder resolves to its index-th .txt/.json (sorted, wrapping)."""
    p = _resolve(path)
    if os.path.isdir(p):
        files = sorted(f for f in os.listdir(p) if f.lower().endswith((".txt", ".json")))
        if not files:
            raise RuntimeError("[JoyLTX StorySource] no .txt / .json files in %s" % p)
        return os.path.join(p, files[int(index) % len(files)])
    return p


def _blocks(path):
    if not os.path.isfile(path):
        raise RuntimeError("[JoyLTX StorySource] not found: %s" % path)
    text = open(path, encoding="utf-8", errors="ignore").read()
    if path.lower().endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("prompts") or data.get("ideas") or data.get("shots") or []
        return [str(x).strip() for x in data if str(x).strip()]
    if _SPLIT.search(text):
        return [b.strip() for b in _SPLIT.split(text) if b.strip()]
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


class JoyLTX_StorySource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": (SOURCES, {"default": SOURCES[0], "tooltip":
                       "Where this run's words come from. scene idea = the box below. idea file = one idea per run, "
                       "the writer expands it. shot file = finished shot prompts, the writer is skipped."}),
            "scene_idea": ("STRING", {"multiline": True, "default": "", "tooltip":
                           "Your premise, used when source = scene idea. Name characters the way their photo folder "
                           "is named and they get cast automatically."}),
            "idea_file": (_entries(), {"tooltip":
                          "File of story ideas from ComfyUI/input (--- on its own line between ideas, or JSON). "
                          "An entry ending in / is the whole folder - INDEX then picks the file. Restart or refresh "
                          "ComfyUI to see files you just added."}),
            "shot_file": (_entries(), {"tooltip":
                          "File of finished shot prompts (--- between shots, or JSON). An entry ending in / is the "
                          "whole folder - INDEX picks the file."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip":
                      "Which idea (or which file, if the path is a folder). Wire a Primitive set to increment and "
                      "queue once per item to work through the file unattended. Wraps at the end."}),
            "refs_attached": ("BOOLEAN", {"default": True, "tooltip":
                              "ON: the writer uses the characters' own names and writes 'looks exactly as in the "
                              "reference photographs' instead of inventing a face - what Refs by Name needs. "
                              "OFF: the writer describes people itself (no reference photos)."}),
        }, "optional": {
            "manual_path": ("STRING", {"default": "", "tooltip":
                            "Optional: a path outside ComfyUI/input (absolute, file or folder). Overrides the dropdown "
                            "for whichever source is selected."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("story_idea", "file_prompts", "use_file", "refs_attached", "name")
    FUNCTION = "read"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Scene idea, idea file or finished shot file - one panel, wired once."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True   # a file that moved is reported by read(), not at queue time

    @classmethod
    def IS_CHANGED(cls, source, scene_idea, idea_file, shot_file, index, refs_attached, manual_path=""):
        try:
            p = _pick_file(manual_path or (idea_file if source.startswith("idea") else shot_file), index)
            return "%s:%s:%s:%s" % (source, p, os.path.getmtime(p), index)
        except Exception:
            return "%s:%s:%s" % (source, scene_idea[:64], index)

    def read(self, source, scene_idea, idea_file, shot_file, index, refs_attached, manual_path=""):
        idea_file = manual_path or idea_file
        shot_file = manual_path or shot_file
        if source.startswith("scene idea"):
            if not str(scene_idea).strip():
                raise RuntimeError("[JoyLTX StorySource] source is 'scene idea' but the box is empty.")
            print("[JoyLTX StorySource] scene idea from the box (%d chars)" % len(scene_idea), flush=True)
            return (scene_idea, "", False, bool(refs_attached), "scene_idea")
        if source.startswith("idea file"):
            p = _pick_file(idea_file, index); b = _blocks(p)
            i = int(index) % len(b)
            print("[JoyLTX StorySource] idea %d/%d from %s" % (i + 1, len(b), p), flush=True)
            return (b[i], "", False, bool(refs_attached), "%s_%02d" % (os.path.splitext(os.path.basename(p))[0], i))
        p = _pick_file(shot_file, index); b = _blocks(p)
        print("[JoyLTX StorySource] %d finished shot prompt(s) from %s (writer skipped)" % (len(b), p), flush=True)
        return ("", json.dumps({"prompts": b}), True, bool(refs_attached), os.path.splitext(os.path.basename(p))[0])


NODE_CLASS_MAPPINGS = {"JoyLTX_StorySource": JoyLTX_StorySource}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_StorySource": "JoyLTX Story Source"}
