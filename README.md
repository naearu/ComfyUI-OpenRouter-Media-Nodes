# ComfyUI OpenRouter Media Nodes

Custom ComfyUI nodes for OpenRouter text, image, video, and text-to-speech generation.

## Setup

Set your API key before starting ComfyUI:

```bash
export OPENROUTER_API_KEY="..."
```

Alternatively, paste the key into each node's `api_key` field.

Generated files are saved under:

```text
ComfyUI/output/openrouter/
```

## Nodes

- `OpenRouter Text`: calls `/api/v1/chat/completions`, returns text and raw JSON.
- `OpenRouter Image`: calls `/api/v1/images`, returns a ComfyUI `IMAGE`, file path, and raw JSON.
- `OpenRouter Video`: calls `/api/v1/videos`, polls the async job, downloads the first completed video, and returns the file path.
- `OpenRouter TTS Audio`: calls `/api/v1/audio/speech`, saves `mp3` or `pcm`, and returns the file path.

Use `provider_json` and `extra_body_json` for model/provider-specific options.

## Model lists

Model dropdowns are loaded from text files in this custom node folder:

- `text_model.txt`
- `image_model.txt`
- `video_model.txt`
- `audio_model.txt`

Write one OpenRouter model id per line. Blank lines and lines starting with `#` are ignored.
