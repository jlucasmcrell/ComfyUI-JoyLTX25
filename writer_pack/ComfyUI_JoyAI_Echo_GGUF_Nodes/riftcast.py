"""RiftCast - portable video-character cartridges (.riftcast), spec v1.0.

Library + CLI. See RIFTCAST_SPEC.md for the format. A .joypack is a plain zip;
this module packs one from a source folder and safely unpacks/materializes one
into the JoyEcho host conventions (joyecho_voices/<tag>/, refs root, loras).

CLI:
    python riftcast.py pack   <source_dir> <out.joypack>
    python riftcast.py inspect <file.joypack>

Security: archives are DATA. No execution, no path traversal (rejected), no
pickle weight formats (safetensors only for loras/).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile

SPEC_VERSION = "1.0"
REQUIRED_KEYS = ("riftcast", "name", "speaker_tag", "voice", "refs", "dna")
VOICE_EXTS = (".mp4", ".wav")
REF_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class JoypackError(ValueError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(entry_name):
    """Reject traversal and absolute paths; return normalized relative path."""
    n = entry_name.replace("\\", "/")
    if n.startswith("/") or ".." in n.split("/") or (len(n) > 1 and n[1] == ":"):
        raise JoypackError(f"unsafe path in archive: {entry_name!r}")
    return n


def pack(source_dir, out_path):
    """Build a .joypack from a folder laid out per the spec.

    If the folder has no manifest.json, a minimal one is synthesized from the
    layout (name = folder name, speaker_tag = lowercased name).
    """
    source_dir = os.path.abspath(source_dir)
    man_path = os.path.join(source_dir, "manifest.json")
    if os.path.isfile(man_path):
        manifest = json.load(open(man_path, encoding="utf-8"))
    else:
        name = os.path.basename(source_dir.rstrip("\\/"))
        manifest = {"riftcast": SPEC_VERSION, "name": name,
                    "speaker_tag": name.lower()}

    # discover required components if unlisted
    def _find(sub, exts):
        d = os.path.join(source_dir, sub)
        if not os.path.isdir(d):
            return []
        return sorted(f"{sub}/{f}" for f in os.listdir(d)
                      if f.lower().endswith(exts))

    if "voice" not in manifest or not manifest["voice"].get("file"):
        vs = _find("voice", VOICE_EXTS)
        if not vs:
            raise JoypackError("no voice anchor found (voice/anchor.mp4|wav)")
        manifest.setdefault("voice", {})["file"] = vs[0]
    if not manifest.get("refs"):
        manifest["refs"] = _find("refs", REF_EXTS)
        if not manifest["refs"]:
            raise JoypackError("no reference stills found (refs/*.png)")
    if not manifest.get("dna"):
        p = "prompts/dna.txt"
        if not os.path.isfile(os.path.join(source_dir, p)):
            raise JoypackError("no DNA text found (prompts/dna.txt)")
        manifest["dna"] = p
    manifest.setdefault("render_law", {"video_fps": 24})

    # collect every real file under the spec dirs + the manifest
    files = []
    for root, _, names in os.walk(source_dir):
        for fn in names:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, source_dir).replace("\\", "/")
            if rel == "manifest.json":
                continue
            if rel.startswith("loras/") and not fn.lower().endswith(".safetensors"):
                raise JoypackError(f"loras/ may only contain .safetensors: {rel}")
            files.append(rel)

    manifest["sha256"] = {manifest["voice"]["file"]:
                          _sha256(os.path.join(source_dir, manifest["voice"]["file"]))}
    for k in REQUIRED_KEYS:
        if k not in manifest:
            raise JoypackError(f"manifest missing required key: {k}")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for rel in files:
            z.write(os.path.join(source_dir, rel), rel)
    return out_path, manifest


def read_manifest(pack_path):
    with zipfile.ZipFile(pack_path) as z:
        try:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        except KeyError:
            raise JoypackError("not a riftcast: manifest.json missing")
    if "riftcast" not in manifest and "joypack" in manifest:
        manifest["riftcast"] = manifest.pop("joypack")  # pre-rename cartridge
    for k in REQUIRED_KEYS:
        if k not in manifest:
            raise JoypackError(f"manifest missing required key: {k}")
    return manifest


def materialize(pack_path, voices_dir, refs_dir, loras_dir, cache_dir):
    """Unpack + install a cartridge into host conventions.

    Returns a dict: {name, speaker_tag, dna, accent_line, rooms, triggers,
                     ltx_loras, zimage_loras, installed_paths}
    """
    manifest = read_manifest(pack_path)
    tag = manifest["speaker_tag"]
    name = manifest["name"]
    out = {"name": name, "speaker_tag": tag, "installed_paths": []}

    with zipfile.ZipFile(pack_path) as z:
        names = {_safe_name(n) for n in z.namelist()}

        def _extract(member, dst_dir, dst_name=None):
            member = _safe_name(member)
            if member not in names:
                raise JoypackError(f"manifest references missing file: {member}")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, dst_name or os.path.basename(member))
            with z.open(member) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            out["installed_paths"].append(dst)
            return dst

        # voice anchor -> joyecho_voices/<tag>/
        vfile = manifest["voice"]["file"]
        want = manifest.get("sha256", {}).get(vfile)
        vdst = _extract(vfile, os.path.join(voices_dir, tag))
        if want and _sha256(vdst) != want:
            os.remove(vdst)
            raise JoypackError("voice anchor failed sha256 verification")

        # refs -> <refs_root>/<NAME>/
        for r in manifest["refs"]:
            _extract(r, os.path.join(refs_dir, name))

        # loras -> models/loras/joypack/<name>/
        out["ltx_loras"], out["zimage_loras"] = [], []
        loras = manifest.get("loras", {})
        for fam, key in (("ltx", "ltx_loras"), ("zimage", "zimage_loras")):
            for entry in loras.get(fam, []) or []:
                dst = _extract(entry["file"], os.path.join(loras_dir, name))
                out[key].append({"path": dst,
                                 "strength": float(entry.get("strength", 1.0)),
                                 "trigger": entry.get("trigger", "")})

        # texts
        out["dna"] = z.read(_safe_name(manifest["dna"])).decode("utf-8").strip()
        out["accent_line"] = manifest.get("voice", {}).get("accent_line", "")
        rooms = manifest.get("environment", {}).get("rooms")
        out["rooms"] = (z.read(_safe_name(rooms)).decode("utf-8").strip()
                        if rooms and _safe_name(rooms) in names else "")
        env_stills = manifest.get("environment", {}).get("stills", []) or []
        for s in env_stills:
            _extract(s, os.path.join(cache_dir, name, "environment"))

    out["render_law"] = manifest.get("render_law", {})
    out["triggers"] = [e.get("trigger", "") for e in
                       (manifest.get("loras", {}).get("zimage", []) or []) if e.get("trigger")]
    return out




def cut(master_path, name, dna_path, out_path=None, speech_at=1.0,
        speech_dur=5.0, ref_times=None, ffmpeg="ffmpeg"):
    """Cut a cartridge directly from a finished render.

    master_path: an mp4 where the character speaks alone with face visible.
    dna_path: the canonical DNA text file (authorial - cannot be automated).
    Cuts a voice anchor at speech_at..+speech_dur and reference stills at
    ref_times (default: three spread frames), then packs.
    """
    import subprocess
    import tempfile
    tag = name.lower()
    ref_times = ref_times or [speech_at + 1.0, speech_at + 3.0, speech_at + 4.5]
    with tempfile.TemporaryDirectory() as td:
        for sub in ("voice", "refs", "prompts"):
            os.makedirs(os.path.join(td, sub))
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", str(speech_at),
                            "-t", str(speech_dur), "-i", master_path,
                            "-c:v", "libx264", "-crf", "18",
                            "-c:a", "aac", "-b:a", "192k",
                            os.path.join(td, "voice", "anchor.mp4")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise JoypackError(f"anchor cut failed: {r.stderr[:200]}")
        for i, t in enumerate(ref_times, 1):
            subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", str(t),
                            "-i", master_path, "-frames:v", "1",
                            os.path.join(td, "refs", f"{tag}_{i:02d}.png")],
                           capture_output=True, text=True)
        shutil.copyfile(dna_path, os.path.join(td, "prompts", "dna.txt"))
        manifest = {"riftcast": SPEC_VERSION, "name": name, "speaker_tag": tag,
                    "display_name": name.title(),
                    "voice": {"file": "voice/anchor.mp4"},
                    "render_law": {"video_fps": 24}}
        json.dump(manifest, open(os.path.join(td, "manifest.json"), "w"), indent=1)
        out_path = out_path or f"{name}.riftcast"
        return pack(td, out_path)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "pack":
        out, man = pack(sys.argv[2], sys.argv[3])
        print(f"packed {man['name']} -> {out}")
        print(f"  speaker_tag={man['speaker_tag']} refs={len(man['refs'])} "
              f"voice={man['voice']['file']}")
    elif cmd == "cut":
        # joypack.py cut <master.mp4> <NAME> <dna.txt> [out.joypack] [speech_at] [speech_dur]
        a = sys.argv
        out, man = cut(a[2], a[3], a[4],
                       out_path=a[5] if len(a) > 5 else None,
                       speech_at=float(a[6]) if len(a) > 6 else 1.0,
                       speech_dur=float(a[7]) if len(a) > 7 else 5.0,
                       ffmpeg=os.environ.get("RIFTCAST_FFMPEG", "ffmpeg"))
        print(f"cut {man['name']} -> {out}")
    elif cmd == "inspect":
        man = read_manifest(sys.argv[2])
        print(json.dumps(man, indent=1))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
