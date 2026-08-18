# ComfyUI_JoyAI_Echo_GGUF_Nodes — modified distribution

## Install

Unzip so that the folder `ComfyUI_JoyAI_Echo_GGUF_Nodes` sits directly inside
`ComfyUI/custom_nodes/`, then restart ComfyUI. If you already have the upstream
pack installed, replace that folder with this one (keep a copy of the old one
if you want to go back).

`README.md` in this folder is **RealRebelAI's original** and documents the pack
itself. This file documents only what was changed.

## Attribution

The original pack is **RealRebelAI's**:
<https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes>

All credit for the node pack, the JoyEcho name and the original design belongs
to them. This is a **modified copy**, redistributed so that the
MiniMax-H3 Multishot workflows actually work as shipped. If you want the
original, unmodified pack, get it from the link above.

## Why this exists

The H3 full workflow drives the prompt writer (`JoyEcho_LLMEnhance`) through
inputs that the upstream release does not have. Installing the upstream pack
leaves the workflow **running but quietly degraded** — it loads, it renders,
and the join rules it depends on are never applied, because the widget that
applies them is not there. That is a bad failure: silent, and it looks like the
chaining simply is not very good.

Rather than ship a workflow that only works on one machine, the modified pack
is here.

## What was changed

Against upstream `nodes.py` (1,182 lines → 4,369):

**On `JoyEcho_LLMEnhance`**
- `join_style` — appends render-verified continuous-scene boundary rules to the
  system prompt (the airlock open, the settled landing, never splitting a line
  across a block, byte-identical scene description). This is the one the H3
  workflow depends on.
- `num_frames` / `fps` — the writer sizes dialogue to the real shot length
  instead of guessing.
- `model_name_custom` — free-text model tag alongside the dropdown.
- `max_tokens` raised to 65536. Thinking models burn 30–40k tokens reasoning;
  16384 cut them mid-thought and returned empty content (measured 2026-08-06).
- Retry on an empty completion.

**Elsewhere in the pack**
- Per-context negative steering, with separate video and audio lanes.
- Memory slot ranges and speaker attribution (`speaker_order`).
- Per-shot head trim, including the shot-1 case.
- Reference scheduling across shots.
- VRAM cache management around the heavy stages.

## What was removed from this distribution

Development scratch only — none of it is upstream's and none is needed to run:
build/dev scripts, editor backups, `__pycache__`, and the `bin/` folder
(193 MB of local binaries).

## Licence

Upstream carries no licence file. This copy is redistributed with attribution
and a clear statement of changes, in the spirit of a publicly released pack. If
RealRebelAI would prefer it not be mirrored, or would rather take these changes
upstream, that is entirely their call and this distribution will be withdrawn on
request — the preferred outcome is that this work lands in their pack.

Contact: raise an issue on the MiniMax-H3 Multishot repo.
