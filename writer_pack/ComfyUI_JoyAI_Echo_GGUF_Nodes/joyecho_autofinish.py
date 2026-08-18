"""JoyEcho_AutoFinish - fire-and-forget master-grade finishing.

Wire: JoyEcho_Generate.images -> AutoFinish.images -> CreateVideo.images
(pure passthrough - the graph's own preview path is unchanged).

When the node executes (i.e. after Generate has written every per-shot
master), it spawns autofinish_worker.py as a DETACHED process and returns
immediately, so the queue item finishes normally. The worker then:
  snapshot shot_XXX.mp4/.wav -> RTXBatchVideoUpscale each via the ComfyUI
  API (jobs run right after this queue item, FIFO) -> pad-wav lossless
  concat -> <name>_<stamp>_MASTER.mp4 next to the shot masters.

Why detached instead of doing the work inline: inline VSR would hold this
queue item open for ~10 extra minutes and block anything queued behind it;
detached, the upscales interleave with the queue like any other job. The
snapshot step makes back-to-back queued renders safe (each finisher pins
its own run's frames before the next render overwrites shot_XXX).

Progress/errors: <output>/joyecho/_autofinish_<stamp>.log - the node
cannot report them (it has already returned by then).

Requires the ComfyUI-RTX-Video-Suite pack (RTXBatchVideoUpscale) and the
NVIDIA MAXINE VSR SDK on this box. On a box without them (e.g. a secondary render box
unless MAXINE is installed) the worker logs the error and exits; the
render itself is unaffected.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import folder_paths
except ImportError:  # outside ComfyUI (tests)
    folder_paths = None


class JoyEcho_AutoFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = pure passthrough, no finishing run."}),
                "master_name": ("STRING", {
                    "default": "JOYECHO",
                    "tooltip": "Final file: <name>_<stamp>_MASTER.mp4"}),
                "scale_factor": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "1.5 on a 1280x736 base -> 1920x1104"}),
                "quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"],
                            {"default": "ULTRA"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 12}),
            },
            "optional": {
                "shots_subdir": ("STRING", {
                    "default": "joyecho",
                    "tooltip": "Output subfolder holding shot_XXX.mp4 "
                               "(matches JoyEcho_Generate output_prefix "
                               "folder)."}),
                "upscale_mode": (["bicubic+cas (deterministic)", "rtx (legacy)"], {
                    "default": "bicubic+cas (deterministic)",
                    "tooltip": "How base-res shots get upscaled for the master. "
                               "bicubic+cas: deterministic ffmpeg resample + "
                               "contrast-adaptive sharpen - zero temporal churn, "
                               "seconds per shot, no GPU (tournament winner, "
                               "2026-07-23). rtx: the old RTXBatchVideoUpscale "
                               "path - sharper synthesis but reshuffles fine "
                               "detail per frame. Ignored when the run produced "
                               "shot_hires masters (those are used directly).",
                }),
                "preview_master": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wait for the background worker and show the REAL "
                               "finished MASTER as this node's canvas preview "
                               "(the in-graph SaveVideo preview is a bit-starved "
                               "re-encode and lies about quality). Adds ~30-90s "
                               "after the render while the master is assembled. "
                               "Auto-disabled in rtx mode - the RTX job queues "
                               "behind this graph item, so waiting would "
                               "deadlock the queue.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "JoyEcho"
    OUTPUT_NODE = True

    def run(self, images, enabled, master_name, scale_factor, quality,
            batch_size, shots_subdir="joyecho",
            upscale_mode="bicubic+cas (deterministic)", preview_master=True):
        if not enabled:
            return (images,)
        out_root = (folder_paths.get_output_directory()
                    if folder_paths else os.path.join(os.getcwd(), "output"))
        shots_dir = os.path.join(out_root, shots_subdir)
        upscaled_dir = os.path.join(out_root, "upscaled_videos")
        worker = str(Path(__file__).resolve().parent / "autofinish_worker.py")

        cmd = [sys.executable, worker,
               "--shots-dir", shots_dir,
               "--upscaled-dir", upscaled_dir,
               "--name", str(master_name).strip() or "JOYECHO",
               "--scale", str(float(scale_factor)),
               "--quality", quality,
               "--batch-size", str(int(batch_size)),
               "--upscale-mode",
               "rtx" if str(upscale_mode).lower().startswith("rtx") else "cas"]
        flags = 0
        if os.name == "nt":
            # CREATE_NO_WINDOW (not DETACHED_PROCESS): the worker gets an
            # INVISIBLE console that its ffmpeg/ffprobe children inherit. A
            # DETACHED worker has no console at all, so every child allocated
            # its own visible one - the flashing focus-stealing popups at the
            # end of each render.
            flags = (subprocess.CREATE_NO_WINDOW
                     | subprocess.CREATE_NEW_PROCESS_GROUP)
        subprocess.Popen(cmd, creationflags=flags,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         close_fds=True)
        print(f"[JoyEcho] AutoFinish: worker spawned for '{master_name}' "
              f"(scale {scale_factor}); upscale jobs will queue after this "
              f"item. Progress: {shots_dir}\\_autofinish_<stamp>.log",
              flush=True)

        # Optionally wait for the worker and preview the REAL master in-canvas.
        # Never in rtx mode: the worker's RTX job queues behind THIS graph item,
        # so blocking here would deadlock the queue. cas/hires routes are
        # self-contained (ffmpeg only) and finish in seconds-to-minutes.
        is_rtx = str(upscale_mode).lower().startswith("rtx")
        if preview_master and not is_rtx:
            import glob as _glob
            import time as _time
            t_spawn = _time.time()
            deadline = t_spawn + 900
            master, stable_size = None, -1
            print("[JoyEcho] AutoFinish: waiting for the master to preview it "
                  "in-canvas (preview_master=True, up to 15 min)...", flush=True)
            while _time.time() < deadline:
                cands = [p for p in _glob.glob(os.path.join(shots_dir, "*_MASTER.mp4"))
                         if os.path.getmtime(p) >= t_spawn - 2]
                if cands:
                    p = max(cands, key=os.path.getmtime)
                    sz = os.path.getsize(p)
                    if sz > 0 and sz == stable_size:
                        master = p
                        break
                    stable_size = sz
                _time.sleep(2)
            if master:
                print(f"[JoyEcho] AutoFinish: master ready -> {master}", flush=True)
                return {"ui": {"images": [{"filename": os.path.basename(master),
                                           "subfolder": shots_subdir,
                                           "type": "output"}],
                               "animated": (True,)},
                        "result": (images,)}
            print("[JoyEcho] AutoFinish: master did not appear within the wait "
                  "window - check the _autofinish log; continuing without the "
                  "in-canvas preview.", flush=True)
        elif preview_master and is_rtx:
            print("[JoyEcho] AutoFinish: preview_master skipped in rtx mode "
                  "(the RTX job runs after this graph item; waiting would "
                  "deadlock the queue).", flush=True)
        return (images,)


NODE_CLASS_MAPPINGS = {"JoyEcho_AutoFinish": JoyEcho_AutoFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_AutoFinish": "JoyEcho Auto-Finish (RTX master)"}
