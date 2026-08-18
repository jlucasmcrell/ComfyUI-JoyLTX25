# ComfyUI JoyAI-Echo GGUF Nodes

<p align="center">
  <img src="https://raw.githubusercontent.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes/main/Screenshot_20260610_113536_Chrome.jpg" alt="JoyAI Echo GGUF ComfyUI Workflow" width="80%">
</p>


**Run JD's JoyAI-Echo (text → video + audio) on consumer hardware.**

A faithful GGUF implementation of [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) for ComfyUI, built so the model can run on an 8 GB GPU with 16 GB system RAM instead of the ~48 GB the full checkpoint expects.

> ⚠️ **Work in progress / experimental.** This is a heavy model on small hardware. Loads and the first encode are slow (see [Performance](#performance)). It works, but it asks for patience.

<!-- Add a preview image at assets/preview.png and it will show here -->
![Preview](assets/preview.png)

---

## What this is

JoyAI-Echo is a modified LTX-2.3 diffusion transformer that generates video **and** matching audio from a text prompt. The full release is a single ~46 GB bf16 checkpoint designed for big GPUs.

This pack makes it runnable on a typical gaming PC by:

- Loading the DiT as a **GGUF** (Q2_K / Q4_K_M) instead of full bf16.
- Building the transformer with JoyAI's own `LTXModelConfigurator` so the architecture is correct (stock ComfyUI GGUF loaders build the **wrong** shapes for this model).
- Running the Gemma-3 text encoder as a **GGUF** (or fp8), memory-mapped on CPU, and freeing it from RAM before the DiT loads — so the encoder and the DiT never sit in memory at the same time. The GGUF encoder drops Stage-1 resident RAM from ~12 GB to ~2.5 GB and roughly halves encode time.
- Streaming DiT blocks to the GPU one at a time during denoising.

All in a single node so you don't have to wire the memory dance yourself.

---

## Requirements

- **ComfyUI** (recent build).
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** (city96) — provides the GGUF tooling this pack relies on.
- A CUDA GPU. **Designed and tested target: RTX 3070, 8 GB VRAM + 16 GB system RAM.** More is better; less will struggle.
- The model files (see [Models](#models)).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes.git
```

Install Python dependencies into your ComfyUI environment (portable users: use the embedded Python):

```bash
python_embeded\python.exe -m pip install -r ComfyUI_JoyAI_Echo_GGUF_Nodes\requirements.txt
```

Restart ComfyUI.

---

## Models


Download from [HuggingFace: realrebelai/JoyAI-Echo_GGUF](https://huggingface.co/realrebelai/JoyAI-Echo_GGUF) and place files like this (standard ComfyUI model folders — `extra_model_paths.yaml` works too):

```
ComfyUI/models/
├── unet/
│   └── JoyAI-Echo-DiT-Q4_K_M.gguf          # the DiT (any quant tier)
├── text_encoders/
│   └── gemma3-12b-joyecho-Q4_K.gguf        # GGUF Gemma encoder (recommended)
│   └── (or gemma_3_12B_it_fp8_scaled.safetensors for the fp8 path)
│   └── joyai_echo_embeddings_processor.safetensors   # the connector
├── vae/
│   ├── joyai_echo_video_vae.safetensors
│   └── joyai_echo_audio_vae.safetensors
└── audio_encoders/
    └── joyai_echo_vocoder.safetensors
```

Pick everything from the dropdowns on the node. `.gguf` encoders are auto-detected — the `gemma_format` setting is ignored for them.

**Quant tiers** (the loader memory-maps the DiT, so a bigger file costs disk + streaming time, *not* resident RAM — pick the largest your disk allows):

| Tier | Size | Quality |
|---|---|---|
| Q2_K | ~9 GB | runs, soft/smeary — proof-of-concept tier |
| Q4_K_M | ~13 GB | **recommended** — sharp, coherent talking heads |
| Q6_K | ~18 GB | closest to bf16 |
| *_RM (Rebels Mix) | +1–2 GB | same base with critical layers boosted to Q8_0 (unsloth-style) |
#

### Architecture config


**Nothing to configure.** Everything the nodes need ships inside this repo:

- `configs/joyai_echo_config.json` — the JoyAI-Echo architecture config (the model is a *modified* LTX-2.3: 9-row scale_shift_table, 4096/2048 connectors — stock GGUF loaders build the wrong shapes without this).
- `gemma_assets/` — the Gemma-3 tokenizer/config sidecars the text encoder needs.

The nodes locate both relative to their own install folder, so they work on any drive, any OS, for any user, with no path field to fill in. There is no config input on the node.
## Usage

Build a minimal graph:

```
RebelsJE_StagedPipeline  →  CreateVideo  →  SaveVideo
```

Select your files in the dropdowns, type a prompt, queue. The node runs three internal stages: **encode → free Gemma → load DiT + VAEs → denoise → decode**, and prints `[StagedJE]` progress lines to the console.

### Key settings


| Setting | What it does |
|---|---|
| `prompt` | What to generate. Quoted speech (`a woman says "..."`) drives the audio. Tip: talking-head training data is full of captioned clips, so the model loves drawing **subtitles** — adding `no subtitles, no captions, no on-screen text` to the prompt helps, and the negative prompt below helps more. |
| `negative_prompt` + `negative_scale` | **Experimental.** The distilled (DMD) pipeline has no CFG, so this can't be true negative guidance; instead it extrapolates the prompt embeddings *away* from the negative in embedding space. `negative_scale = 0` disables it completely. Start around `0.3–0.8`; too high degrades the image. Default negative targets the subtitle habit. |
| `num_frames` | Must be `1 + multiple of 8` (25, 49, 121...). Other values are rounded down. |
| `video_height/width` | Multiples of 32 (others round down). The model badly wants real resolution — 512×768 looks dramatically better than tiny test sizes. |
| `gemma_format` | Only used for `.safetensors` encoders. Ignored for `.gguf` (auto-detected). |
| `encode_on_cpu` | Keep **on** for 8 GB GPUs. |
| `sequential_offload` | Keep **on** for 8 GB GPUs — streams one DiT block to the GPU at a time. |

**Caching:** the node caches conditioning per prompt text and the built model between runs. Re-running with a new seed skips both the Gemma encode and the model load — iteration runs start denoising almost immediately.
## Performance

Be realistic about an 8 GB / 16 GB box:

- **Text encode** runs a 12 B model on CPU. Expect **several minutes**, with one CPU core pegged and ~12 GB RAM in use. It looks like a hang — it isn't.
- **DiT load** reads several GB from disk. Put your models on an **SSD**; a spinning HDD makes this painful.
- **Denoising** streams blocks to the GPU and is slow per step at this quantization. Keep frames/resolution low while testing.

Watch the console for `[Rebels JE] DiT GGUF swap: matched N/N Linear layers` — that confirms the DiT stayed quantized in RAM (the whole point). If it ever prints `0/N`, something's pointed at the wrong file.

---

## Troubleshooting


- **`paging file is too small` (os error 1455)** — Windows ran out of commit. Check free space on the drive holding your pagefile, and use the GGUF Gemma encoder (it needs ~2.5 GB resident instead of ~12 GB).
- **Subtitles burned into the video** — known model habit from captioned training data. Add `no subtitles, no captions, no on-screen text` to the positive prompt and/or use the negative prompt with `negative_scale` ≈ 0.5.
- **Audio sounds deep / slow / demonic** — you're on an old version of these nodes; the current ones always label the waveform with the vocoder's native sample rate. Update.
- **Second run hangs at "Sequential offloading installed"** — old-version bug (offloader hooks stacked on the cached model). Update.
- **Output dimensions differ from what you set** — height/width round down to multiples of 32 and frames to `1 + 8k`; the model requires it.
## Acknowledgements

- [JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) — Echo Team @ Joy Future Academy, JD
- [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — Lightricks
- [Gemma-3](https://huggingface.co/google/gemma-3-12b-it) — Google
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — city96

Forked from [zhuang2002/ComfyUI_JoyAI_Echo](https://github.com/zhuang2002/ComfyUI_JoyAI_Echo).

---

## License

For academic research and non-commercial use only, following the upstream JoyAI-Echo license. Gemma-3 components are subject to [Google's Gemma license](https://ai.google.dev/gemma/terms); review those terms before redistributing any Gemma files.
