# Install

1. ComfyUI 0.32 or newer.
2. Manager → Install via Git URL → https://github.com/jlucasmcrell/ComfyUI-JoyLTX25 (or unzip the release into `custom_nodes/ComfyUI-JoyLTX25`).
3. Copy `writer_pack/ComfyUI_JoyAI_Echo_GGUF_Nodes/` from this zip into `custom_nodes/` (the writer; it is RealRebelAI's pack - https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes - with our long-story / extend-take / reference writing rules bundled; the Manager copy does not have them). For `.gguf` DiT files also install **ComfyUI-GGUF**.
4. Models:
   - DiT: one file from https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf (RTX 30/40) or https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native (RTX 50) → `models/diffusion_models/`
     16 GB: https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-DiT-Q4_K_S.gguf or https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-DiT-comfy-w4a8.safetensors
     24 GB: https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-DiT-Q5_K_M.gguf or https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-DiT-comfy-mix4x8-17.0GB.safetensors
     32 GB: https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-gguf/blob/main/LTX25dist-echoVid-070T30-DiT-Q8_0.gguf or https://huggingface.co/joeygambino/joyai-echo-ltx25-echoVid-comfy-native/blob/main/LTX25dist-echoVid-070T30-DiT-comfy-int8.safetensors
     (also on Civitai: Joy-LTX 2.5, being uploaded now)
   - `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-audio-vae-bf16.safetensors` → `models/vae/` (Lightricks/LTX-2.5)
   - `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` → `models/latent_upscale_models/` (Lightricks/LTX-2.5); optional `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` (Lightricks/LTX-2.3)
   - text encoder `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (Lightricks/LTX-2.5) or the 10.6 GB `gemma4-12b-ltx25-comfy-w4a8.safetensors` (joeygambino/LTX-2.5-Quantized) → `models/text_encoders/`
5. Restart ComfyUI. Open `custom_nodes/ComfyUI-JoyLTX25/workflows/JoyLTX25_Take.json` (or drag it onto the canvas). Pick your files in the loaders. Type a premise. Run.
6. Writer: point the JoyEcho LLMEnhance node at an Ollama (`http://localhost:11434/v1`) with a cloud model, or any OpenAI-compatible endpoint. No writer? mode = passthrough and paste your own prompt.
