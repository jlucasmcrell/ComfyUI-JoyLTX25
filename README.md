# ComfyUI-JoyLTX25 — Joy-LTX 2.5

**JoyAI-Echo's performance on LTX-2.5.** Two canvases and the four small nodes they need.

- **JoyLTX25_Take** — one prompt, one take, one generation (8–30 s). A planner sizes the two
  render passes to *your* card (x2 / x1.5 / single pass) so it just runs.
- **JoyLTX25_Multishot** — one story, several shots, one video. Each shot is one generation
  (VRAM = one shot), joined by an **AV-extend**: the previous shot's last frames *and sound* are
  pinned into the next one, so a take can be as long as you like with no visible seam, or you
  can cut between angles and keep the same voice.
- **JoyLTX25_Take_PLUS / JoyLTX25_Multishot_PLUS** — the same two canvases with four more lanes:
  a **LoRA stack** (4 slots), **Refs by Name** (cast characters from `input/joyecho_refs/<name>/`
  photo folders — one photo per character, and from their second shot on their own rendered face
  and voice carry into every later shot), **Prompts from File** (your own shot list), and **audio
  export** (flac). Start with the plain canvases; move to PLUS when you want any of those.

Models: [GGUF](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf) (RTX 30/40) ·
[comfy-native](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native) (RTX 50).
Both doses in each: **070T30** (default, cleaner skin) and **100T50** (livelier performances).
Pick by card: 16 GB [Q4_K_S](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-v2-DiT-Q4_K_S.gguf) / [w4a8](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-v2-DiT-comfy-w4a8.safetensors) · 24 GB [Q5_K_M](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-v2-DiT-Q5_K_M.gguf) / [mix4x8-17.0GB](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-v2-DiT-comfy-mix4x8-17.0GB.safetensors) · 32 GB [Q8_0](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-v2-DiT-Q8_0.gguf) / [int8](https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-v2-DiT-comfy-int8.safetensors) (100T50 = same links with `100T50` in the name; **v2** files = LTX-2.5 dev + the JoyAI-Echo delta with the official distilled LoRA baked at 0.5 - the v1 files were over-cooked and are gone). Dev merges (bring your own distill LoRA): https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-dev · Also on Civitai: **Joy-LTX 2.5**.

## Install (5 minutes)

1. Manager → Install via Git URL → `https://github.com/jlucasmcrell/ComfyUI-JoyLTX25` (or unzip into `custom_nodes/`).
2. Copy `writer_pack/ComfyUI_JoyAI_Echo_GGUF_Nodes/` from this zip into `custom_nodes/` (the LLM writer that turns your premise into shot prompts - RealRebelAI's pack with our writing rules bundled; the Manager version lacks them). For `.gguf` models also install **ComfyUI-GGUF**.
3. Put a Joy-LTX DiT file in `models/diffusion_models/`. 24 GB → `…070T30-DiT-Q5_K_M.gguf`; 32 GB → `…070T30-DiT-comfy-int8.safetensors`; 16 GB → `…Q4_K_S.gguf` or `…comfy-w4a8`.
4. From [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5): the two VAEs → `models/vae/`, the x2 latent upscaler → `models/latent_upscale_models/`, the gemma-4 text encoder (`…comfy-int8-convrot`, or the 10.6 GB w4a8 from [LTX-2.5-Quantized](https://huggingface.co/joeygambino/LTX-2.5-Quantized) for 16 GB) → `models/text_encoders/`.
   Optional: `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` from [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) → `models/latent_upscale_models/` (the planner's x1.5 step for 24 GB cards).
5. Restart ComfyUI, open `workflows/JoyLTX25_Take.json`, pick your files in the three loaders, type a premise, Run.

## The two canvases

**Take**: SCENE IDEA → writer (JoyEcho LLMEnhance, hosted model recommended) → JoyLTX Script folds
the beats into one LTX prompt → pass 1 at the render size → latent upscale → pass 2 → decode.
`JoyLTX Take Controls` = render size, seconds, upscale. `auto` reads your VRAM and prints what it
planned; `x2` / `x1.5` / `none` override it (stream and wait if you like), `vram_gb` steers auto. **USE I2V** (off by default): load a picture, flip the gate, and the take
starts from that frame (image-to-video) instead of text only. 8 s at 960x544 auto = x2 → 1920x1088 on 24 GB in ~7 min (RTX 3090),
~2 min on an RTX 5090.

**Multishot**: SCENE IDEA → writer (its `num_shots` = how many shots) writes N shot prompts → `JoyLTX Multishot Sampler`
renders each shot with the same two-pass pipeline and joins them:
- `continue` — seamless take across generations (writer join style *extend take*).
- `cut` — only the sound is pinned: the voice carries across a picture cut (writer *cuts* / *continuous scene with cuts*).
- `fresh` — independent shots.
`overlap` = pinned latent frames (3 = 0.7 s, trimmed on output). `save_every_shot` writes each raw
shot to `output/video/JOYLTX_SHOTS/`.

**Same people in every shot, without reference photos:** `identity_ref` attaches a frame of shot 1
to every later shot as an in-context keyframe reference (LTX-2.5 AddGuide - it conditions, it never
appears), so a cut to a new angle keeps the same face, clothes and room; the voice is carried by the
audio pin. `identity_strength` 0.4-0.7 = same person, free composition; 1.0 = near first-frame copy.
Optional `shot_images` (one per shot) or the **USE I2V** switch (a Load Image for the first frame of
shot 1) if you want to start from your own picture.

## Numbers that matter

| card | Take, 8 s 960x544 auto | Multishot, per 8 s shot |
|---|---|---|
| RTX 5090 (int8) | ~2 min (x2 → 1920x1088) | ~2 min |
| RTX 3090 (Q5_K_M) | ~7 min (x2) · ~8 min x1.5 at 1280x736 | ~4 min (x1.5) |
| 30 s single pass 1280x736 | 5 min (5090) · 17 min (3090) | — |

Two-pass reads better than a same-token single pass; on 24 GB `auto` picks x2 for 8 s clips and
x1.5 for 12 s at 960x544. Fewer, longer shots mean fewer joins: 4 × 30 s single pass on 24 GB is a
2-minute take with three joins. Cloud writer models sometimes return 503/empty — pick another in
the writer's list and rerun.

## Nodes

`JoyLTX Take Controls` · `JoyLTX Multishot Sampler` · `JoyLTX Script` · `JoyLTX Any Switch`
(lazy A/B) · `JoyLTX Optional Image` (the I2V / END-frame gates: emit nothing when off) ·
`JoyLTX Keyframes` (end frame and mid keyframes for a take; the sampler has `end_images` per shot). Plus an in-process patch that fixes ComfyUI's x1.5 latent upsampler when it is only
partially loaded (a real crash on 24 GB cards). See SETTINGS.md.

## Credits / license

JoyAI-Echo by JD · LTX-2.5 by Lightricks · merge, quants, nodes, canvases by joeygambino.
MIT for this pack; the model files follow the LTX-2.x Community License.
