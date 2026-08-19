# Changelog

## 1.2.0
- **Story Source panel** on the PLUS canvases: one node picks where a run's words come from - your scene idea, a file of story IDEAS walked one per queue by an INDEX primitive, or finished shot prompts that skip the writer. File pickers are dropdowns of everything under `ComfyUI/input`, folder entries walk a folder, and an optional `manual_path` reaches outside. Replaces the loose Prompts-from-File / two booleans / extra switch.
- **refs_attached** is wired from that panel: the writer then uses each character's own name and writes "looks exactly as in the reference photographs" instead of inventing a face - what Refs by Name needs to cast anyone.
- **Refs by Name**: `refs_root` is a dropdown of the character libraries found under `ComfyUI/input`; new `crop` (portrait / full photo) keeps the person and drops the room, so a scene photo no longer drags its set into the shot; `pick` defaults to one photo per character for the whole piece; `[ref: name]` in a shot forces a pick and is stripped before rendering.
- **Multishot sampler**: per-character identity and voice lock - from a character's second shot on, their own rendered face (face-area crop of their first shot) and their own audio tail are carried into every later shot of theirs, instead of the previous shot's.
- Canvas notes for Story Source and Refs by Name; shorter node titles so nodes size to their lane.

## 1.1.0
- PLUS canvases: LoRA stack (4 slots), Refs by Name, Prompts from File, audio export (flac).
- `JoyLTX Load Model` takes `.safetensors` or `.gguf` and resolves a moved file by name.
- v2 model names and links throughout.

## 1.0.0
- First release: Take and Multishot canvases, VRAM planner, AV-extend joins, identity anchor, keyframes, image folders, bundled writer.
