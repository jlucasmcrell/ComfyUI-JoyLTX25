# Settings reference (ComfyUI-JoyLTX25)

## JoyLTX Take Controls
- **width / height** – render size of pass 1 (multiples of 32). 960x544 ships. Output = this x2 or x1.5 when two passes fit, else this size.
- **take_seconds** – length of the take (Take canvas) or seconds PER SHOT (Multishot canvas). Frames = 8n+1 at 24 fps.
- **beat_seconds** – how the writer splits speech into beats (Take canvas only). Shorter beats = less dead air.
- **upscale** – `auto` plans for your card: x2 → x1.5 → single pass, whichever fits (~2200 pass-2 tokens per GB of VRAM), and prints the plan. `x2` / `x1.5` / `none` are hard overrides: the planner obeys and warns "FORCED, expect streaming or an OOM" – if you accept the wait, that is your mode.
- **vram_gb** – 0 = detect. Type a number to steer `auto` (48 on a 24 GB card = plan as if you had 48 and stream), or a smaller one if other things share the card.
- Outputs: pass1_width, pass1_height, frames, two_pass (bool), use_x15 (bool: drives the x2/x1.5 upscaler switch), beats, beat_frames, summary.

## JoyLTX Multishot Sampler
- **prompts** – the writer's `{"prompts": [...]}` JSON (wired) or paste blocks separated by `---`.
- **width / height / frames_per_shot** – per shot (wired from Take Controls in the canvas).
- **shot_count** – 0 = all the writer's prompts.
- **join** – `continue` (video + audio tail pinned: seamless), `cut` (audio only: voice carries, new picture), `fresh`.
- **overlap** – latent frames pinned from the previous shot (3 = 17 px frames = 0.7 s, trimmed on output). More = smoother join, less new content per shot.
- **seed / seed_per_shot** – seed_per_shot varies the seed by shot (recommended).
- **sampler_name / sigmas_pass1 / sigmas_pass2** – the distilled schedules ship (8 + 3 steps, euler_ancestral). Leave them.
- **two_pass** – upscale + refine each shot (needs upscale_model). Both passes pin the previous tail, so the refine never redraws the join.
- **video_cfg / audio_cfg** – 1.0 for the distilled model.
- **save_every_shot** – raw shots to `output/video/JOYLTX_SHOTS/`.
- **identity_ref** – `off` / `cuts only: frame from shot 1` (default) / `all shots: frame from shot 1` / `cuts only: last frame of previous shot`. Attaches that frame to later shots as an in-context keyframe reference so the same person appears in every shot without reference photos. `all shots` also anchors continue mode to shot 1's look.
- **identity_strength** – attention weight of that reference (0.6 ships). 0.4-0.7 keeps identity with free composition; 1.0 is close to copying the first frame.
- **start_image / shot_images / image_strength** – first frame for shot 1 (the USE I2V gate on the canvas) / one image per shot (cut & fresh modes) / how hard to hold it.

## JoyLTX Script
Folds the writer's shot list into ONE LTX prompt (scene stated once, beats follow). Outputs: shots_script (passthrough), ltx_prompt, shot_count.

## JoyLTX Any Switch
Lazy A/B: only the selected branch executes (the other loader never loads).

## Writer (JoyEcho LLMEnhance, from the JoyAI-Echo pack)
mode `long_story (multi-shot)`; join style `extend take` for continue mode, `cuts` or `continuous scene with cuts` for cut mode; hosted models (deepseek-v4-pro / glm-5.2 / minimax-m3 via Ollama cloud) write the best beats; `passthrough` lets you paste your own JSON.

## JoyLTX Optional Image
`enabled` off = emits nothing (true text-to-video; the Load Image upstream is skipped). On = passes the image and a BOOLEAN that flips the Take canvas's video-latent switch to image-to-video.

## JoyLTX Keyframes (Take canvas)
Optional END frame (the USE END FRAME gate) and/or MID keyframes (image batch + `mid_frame_indices`, e.g. `48,96`). Images become LTX-2.5 keyframe tokens (they condition, they never appear); `strength` 0.8 ships (1.0 = reproduce the frame). The `crop keyframes after pass 1` node removes them before the upscale/decode - leave it in place. With nothing connected both nodes pass through at no cost.

## Multishot: end_images / keyframe_strength
One image per shot = keyframe at that shot's last frame (first->last-frame shots when combined with shot_images or identity_ref). Cropped after pass 1 automatically.


## PLUS canvases

- **JoyLTX LoRA Stack** – four slots on the DiT, each a LoRA file + strength; `(none)` or 0 = skipped. Model-only. On a DEV merge the official `ltx-2.5-22b-distilled-lora-450` at 0.4-0.6 makes it few-step.
- **JoyLTX Refs by Name** – `refs_root` (default `joyecho_refs` under ComfyUI/input) holds one folder per character; the folder name is the name you write in the prompt. `pick` = one per character (default: the same photo in every shot), first, random per shot, random by seed. Outputs one photo per shot + a mask of character names + the cleaned prompts (`[ref: name]` markers stripped). Use portraits or clean character shots - a busy scene photo drags its set into the shot.
- **Multishot sampler → ref_images / ref_mask / ref_strength** – each shot's photo is attached at frame 0 (appended, cropped after pass 1) at `ref_strength` (0.5). With character names in the mask, the sampler locks each character from their second shot on: their own rendered frame from their first shot at `identity_strength` plus the photo at half strength (visual lock), and in cut mode their own audio tail is pinned instead of the previous shot's (voice lock).
- **Keyframes → ref_image / ref_strength** (Take) – the same frame-0 photo reference for a single take.
- **JoyLTX Prompts from File** – `path` under ComfyUI/input (or absolute): `---`-separated text or `{"prompts": [...]}` JSON → the sampler's prompts.
- **save audio** – core SaveAudio (flac) on the finished track.
