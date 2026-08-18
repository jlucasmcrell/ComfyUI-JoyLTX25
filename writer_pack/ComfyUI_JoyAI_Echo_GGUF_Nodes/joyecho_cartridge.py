"""JoyEcho_CartridgeLoader - load a .joypack character cartridge.

Materializes the cartridge into the host's existing conventions (voice anchor
into input/joyecho_voices/<tag>/, reference stills into the refs root, LoRAs
into models/loras/joypack/<name>/) and outputs the character's texts for
prompt assembly. The render path needs no changes - after loading, any script
whose speaker tag matches is voiced and identified as the cartridge character.

See RIFTCAST_SPEC.md for the format.
"""
import json
import os

try:
    import folder_paths
except Exception:
    folder_paths = None

from .riftcast import materialize, read_manifest, JoypackError


def _packs_dir():
    if folder_paths:
        base = folder_paths.get_input_directory()
    else:
        base = os.path.join(os.getcwd(), "input")
    d = os.path.join(base, "riftcast")
    os.makedirs(d, exist_ok=True)
    return d


class JoyEcho_CartridgeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        packs = ["(none)"]
        try:
            packs += sorted(f for f in os.listdir(_packs_dir())
                            if f.lower().endswith((".riftcast", ".joypack")))
        except Exception:
            pass
        return {
            "required": {
                "cartridge": (packs, {
                    "tooltip": "A .riftcast cartridge from input/riftcast/. Loading installs "
                               "the character's voice anchor, reference stills and "
                               "LoRAs into this workflow's normal folders - any "
                               "script whose speaker tag matches renders as this "
                               "character. See RIFTCAST_SPEC.md."}),
                "refs_root": ("STRING", {"default": "joyecho_refs", "tooltip":
                              "Where reference stills install (RefPicker's root). "
                              "Absolute path or relative to the input folder."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("speaker_tag", "dna_text", "accent_line", "rooms_text", "report")
    FUNCTION = "load"
    CATEGORY = "JoyAI-Echo"

    def load(self, cartridge, refs_root="joyecho_refs"):
        if cartridge == "(none)":
            return ("", "", "", "", "no cartridge selected")
        pack_path = os.path.join(_packs_dir(), cartridge)

        in_dir = (folder_paths.get_input_directory() if folder_paths
                  else os.path.join(os.getcwd(), "input"))
        voices_dir = os.path.join(in_dir, "joyecho_voices")
        refs_dir = (refs_root if os.path.isabs(refs_root)
                    else os.path.join(in_dir, refs_root))
        if folder_paths:
            loras_dir = os.path.join(folder_paths.models_dir, "loras", "riftcast")
        else:
            loras_dir = os.path.join(os.getcwd(), "models", "loras", "riftcast")
        cache_dir = os.path.join(in_dir, "riftcast", "_cache")

        try:
            res = materialize(pack_path, voices_dir, refs_dir, loras_dir, cache_dir)
        except (JoypackError, OSError) as e:
            raise ValueError(f"riftcast load failed: {e}")

        law = res.get("render_law", {})
        law_note = ""
        if law.get("video_fps"):
            law_note = (f" | render law: video_fps={law['video_fps']} "
                        f"(authored under the 24fps accent law)" if law["video_fps"] == 24
                        else f" | render law: video_fps={law['video_fps']}")
        loras_note = ""
        if res["ltx_loras"]:
            loras_note = " | LTX loras: " + ", ".join(
                f"{os.path.basename(l['path'])}@{l['strength']}" for l in res["ltx_loras"])
        report = (f"loaded '{res['name']}' as speaker '{res['speaker_tag']}' - "
                  f"voice anchor + {sum(1 for p in res['installed_paths'] if refs_dir in p)} "
                  f"ref(s) installed{loras_note}{law_note}")
        print(f"[JoyEcho] Cartridge: {report}", flush=True)

        dna = res["dna"]
        if res.get("triggers"):
            dna = dna  # triggers are for Z-Image prompts; reported, not injected
        return (res["speaker_tag"], dna, res.get("accent_line", ""),
                res.get("rooms", ""), report)


NODE_CLASS_MAPPINGS = {"JoyEcho_CartridgeLoader": JoyEcho_CartridgeLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_CartridgeLoader": "RiftCast Loader (.riftcast)"}


def auto_materialize_all():
    """Unpack any new/changed .joypack in input/joypacks/ at startup.

    Makes cartridges drop-in: put the file in the folder, restart (or load
    once via the node), and the character is castable by speaker tag. A
    content-hash marker in _cache/ keeps this a no-op for already-installed
    packs, so startup cost is one hash per cartridge.
    """
    import hashlib
    try:
        d = _packs_dir()
        in_dir = (folder_paths.get_input_directory() if folder_paths
                  else os.path.join(os.getcwd(), "input"))
        marker_dir = os.path.join(d, "_cache", "_installed")
        os.makedirs(marker_dir, exist_ok=True)
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".riftcast", ".joypack")):
                continue
            path = os.path.join(d, fn)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
            marker = os.path.join(marker_dir, fn + "." + h)
            if os.path.isfile(marker):
                continue
            try:
                res = materialize(
                    path,
                    voices_dir=os.path.join(in_dir, "joyecho_voices"),
                    refs_dir=os.path.join(in_dir, "joyecho_refs"),
                    loras_dir=(os.path.join(folder_paths.models_dir, "loras", "riftcast")
                               if folder_paths else
                               os.path.join(os.getcwd(), "models", "loras", "riftcast")),
                    cache_dir=os.path.join(d, "_cache"))
                open(marker, "w").write(h)
                print(f"[JoyEcho] Cartridge auto-installed: '{res['name']}' as "
                      f"speaker '{res['speaker_tag']}' ({fn})", flush=True)
            except Exception as e:
                print(f"[JoyEcho] Cartridge auto-install FAILED for {fn}: {e}",
                      flush=True)
    except Exception as e:
        print(f"[JoyEcho] Cartridge auto-install scan skipped: {e}", flush=True)


auto_materialize_all()
