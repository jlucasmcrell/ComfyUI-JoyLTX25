"""JoyLTX_LoraStack: up to four LoRAs on the model in one node, each with "(none)" and a strength.

Slots set to "(none)" or strength 0 cost nothing. Uses ComfyUI's own LoRA loader underneath (LoraLoaderModelOnly),
so anything that loads there loads here - LTX-2.5 character / style LoRAs, or the official distilled LoRA on a
DEV merge (0.4-0.6). Model-only: LTX LoRAs do not touch the Gemma text encoder.

JoyLTX_PromptFile: read shot prompts from a text / json file under ComfyUI/input (or an absolute path) - one
prompt per line-block separated by --- lines, or {"prompts": [...]} - so an episode written outside ComfyUI
(or by a previous run) drives the sampler without pasting.
"""
import os
import folder_paths
import nodes


def _lora_names():
    try:
        return ["(none)"] + list(folder_paths.get_filename_list("loras"))
    except Exception:
        return ["(none)"]


class JoyLTX_LoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        names = _lora_names()
        req = {"model": ("MODEL",)}
        for i in range(1, 5):
            req["lora_%d" % i] = (names, {"default": "(none)"})
            req["strength_%d" % i] = ("FLOAT", {"default": 1.0 if i == 1 else 0.0, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": req}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Four LoRA slots (model only). (none) / strength 0 = skipped. Distilled LoRA on a DEV merge: 0.4-0.6."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True   # a saved name that moved is reported in apply(), not at queue time

    def apply(self, model, **kw):
        loader = nodes.LoraLoaderModelOnly()
        applied = []
        for i in range(1, 5):
            name = kw.get("lora_%d" % i, "(none)")
            s = float(kw.get("strength_%d" % i, 0.0))
            if not name or name == "(none)" or abs(s) < 1e-6:
                continue
            if folder_paths.get_full_path("loras", name) is None:
                raise RuntimeError("[JoyLTX LoraStack] LoRA file not found: %r (pick it again in slot %d)" % (name, i))
            model = loader.load_lora_model_only(model, name, s)[0]
            applied.append("%s @ %.2f" % (os.path.basename(name), s))
        print("[JoyLTX LoraStack] " + (", ".join(applied) if applied else "no LoRAs (all slots none/0)"), flush=True)
        return (model,)


class JoyLTX_PromptFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": "joyltx_prompts/episode.txt", "tooltip":
                     "Relative to ComfyUI/input, or absolute. .txt: prompts separated by --- lines (or one per "
                     "paragraph if there are no ---). .json: {\"prompts\": [...]} or a list."}),
        }, "optional": {
            "reload": ("BOOLEAN", {"default": True, "tooltip": "Re-read the file every run (turn off to cache)."}),
            "mode": (["file (path)", "folder + index"], {"default": "file (path)", "tooltip":
                     "file = the path above. folder + index = the N-th prompt file (sorted) in `folder`; wire `index` "
                     "from a Primitive set to increment and queue the graph N times to render a folder of episodes unattended."}),
            "folder": ("STRING", {"default": "joyltx_prompts", "tooltip": "Folder under ComfyUI/input (or absolute) holding .txt / .json prompt files."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Which file (0-based, wraps). Wire a Primitive on 'increment' to walk the folder."}),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompts", "shot_count", "name")
    FUNCTION = "read"
    CATEGORY = "JoyLTX"
    DESCRIPTION = "Shot prompts from a file (--- separated text or JSON) -> the sampler's prompts input."

    @classmethod
    def IS_CHANGED(cls, path, reload=True, mode="file (path)", folder="joyltx_prompts", index=0):
        p = cls._pick(path, mode, folder, index)
        try:
            return ("%s:%s" % (p, os.path.getmtime(p))) if reload else p
        except Exception:
            return p

    @classmethod
    def _pick(cls, path, mode, folder, index):
        if str(mode).startswith("folder"):
            d = cls._resolve(folder)
            files = sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.lower().endswith((".txt", ".json")))
            if not files:
                return os.path.join(d, "(no .txt/.json files)")
            return os.path.join(d, files[int(index) % len(files)])
        return cls._resolve(path)

    @staticmethod
    def _resolve(path):
        p = str(path or "").strip().strip('"')
        if not os.path.isabs(p):
            p = os.path.join(folder_paths.get_input_directory(), p)
        return p

    def read(self, path, reload=True, mode="file (path)", folder="joyltx_prompts", index=0):
        import json, re
        p = self._pick(path, mode, folder, index)
        if not os.path.isfile(p):
            raise RuntimeError("[JoyLTX PromptFile] not found: %s" % p)
        text = open(p, encoding="utf-8", errors="ignore").read()
        prompts = None
        if p.lower().endswith(".json"):
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("prompts") or data.get("shots") or []
            prompts = [str(x) for x in data]
        else:
            if re.search(r"^\s*---+\s*$", text, re.M):
                prompts = [s.strip() for s in re.split(r"^\s*---+\s*$", text, flags=re.M) if s.strip()]
            else:
                prompts = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
        name = os.path.splitext(os.path.basename(p))[0]
        print("[JoyLTX PromptFile] %d prompt(s) from %s%s" % (len(prompts), p, (" (index %d)" % int(index)) if str(mode).startswith("folder") else ""), flush=True)
        return (json.dumps({"prompts": prompts}), len(prompts), name)


NODE_CLASS_MAPPINGS = {"JoyLTX_LoraStack": JoyLTX_LoraStack, "JoyLTX_PromptFile": JoyLTX_PromptFile}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyLTX_LoraStack": "JoyLTX LoRA Stack",
                              "JoyLTX_PromptFile": "JoyLTX Prompts from File"}
