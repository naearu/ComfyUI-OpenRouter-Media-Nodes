import base64
import io
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import folder_paths


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_SUBDIR = "openrouter"
DEFAULT_TIMEOUT = 600
DEFAULT_TEXT_TIMEOUT = 60
COMPONENT_DIR = Path(__file__).resolve().parent
MODEL_FILE_FALLBACKS = {
    "text_model.txt": ["openai/gpt-4o-mini"],
    "image_model.txt": ["openai/gpt-image-1"],
    "video_model.txt": ["google/veo-3.1"],
    "audio_model.txt": ["openai/gpt-4o-mini-tts-2025-12-15"],
}


def _model_choices(file_name: str) -> list[str]:
    path = COMPONENT_DIR / file_name
    fallback = MODEL_FILE_FALLBACKS[file_name]
    if not path.exists():
        return fallback

    choices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        choices.append(value)
    return choices or fallback


def _headers(api_key: str) -> dict[str, str]:
    key = (api_key or "").strip() or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OpenRouter API key is required. Set OPENROUTER_API_KEY or fill api_key.")

    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/comfyanonymous/ComfyUI",
        "X-Title": "ComfyUI OpenRouter Media Custom Node",
    }


def _raise_for_response(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    raise RuntimeError(f"OpenRouter API error {response.status_code}: {payload}")


def _output_dir() -> Path:
    path = Path(folder_paths.get_output_directory()) / OUTPUT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_ext_from_mime(media_type: str | None, fallback: str) -> str:
    if not media_type:
        return fallback
    if media_type == "image/svg+xml":
        return ".svg"
    ext = mimetypes.guess_extension(media_type.split(";")[0].strip())
    return ext or fallback


def _unique_path(prefix: str, ext: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _output_dir() / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}{ext}"


def _parse_json_object(value: str, field_name: str) -> dict[str, Any] | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{field_name} must be a JSON object.")
    return parsed


def _tensor_to_data_urls(image, image_format: str = "png") -> list[str]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow and numpy are required to encode ComfyUI IMAGE references.") from exc

    if image is None:
        raise RuntimeError("image reference is missing.")

    images = image if getattr(image, "ndim", None) == 4 else image[None,]
    urls = []
    for img in images:
        arr = img.detach().cpu().numpy()
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil.save(buffer, format=image_format.upper())
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        urls.append(f"data:image/{image_format};base64,{encoded}")
    return urls


def _image_reference_payloads(image, image_format: str = "png") -> list[dict[str, Any]]:
    return [{"type": "image_url", "image_url": {"url": url}} for url in _tensor_to_data_urls(image, image_format)]


def _b64_image_to_tensor(b64_json: str):
    try:
        import numpy as np
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow, numpy, and torch are required to return ComfyUI IMAGE tensors.") from exc

    raw = base64.b64decode(b64_json)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(pil).astype("float32") / 255.0
    return torch.from_numpy(arr)[None,]


def _chat_text(payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    _raise_for_response(response)
    return response.json()


class OpenRouterText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_choices("text_model.txt"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 200000}),
                "timeout_seconds": ("INT", {"default": DEFAULT_TEXT_TIMEOUT, "min": 5, "max": 600}),
                "api_key": ("STRING", {"default": "", "password": True}),
            },
            "optional": {
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
                "extra_body_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "OpenRouter"

    def generate(
        self,
        model,
        prompt,
        system_prompt,
        temperature,
        max_tokens,
        timeout_seconds,
        api_key,
        provider_json="",
        extra_body_json="",
    ):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        provider = _parse_json_object(provider_json, "provider_json")
        if provider:
            payload["provider"] = provider
        extra = _parse_json_object(extra_body_json, "extra_body_json")
        if extra:
            payload.update(extra)

        try:
            data = _chat_text(payload, api_key, timeout_seconds)
        except requests.Timeout as exc:
            raise RuntimeError(
                f"OpenRouter text request timed out after {timeout_seconds}s. "
                "Choose a faster model/provider, reduce max_tokens, or increase timeout_seconds."
            ) from exc
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (text or "", json.dumps(data, ensure_ascii=False, indent=2))


class OpenRouterImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_choices("image_model.txt"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": ("STRING", {"default": "1024x1024"}),
                "aspect_ratio": ("STRING", {"default": "auto"}),
                "resolution": ("STRING", {"default": ""}),
                "quality": (["auto", "low", "medium", "high"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": "", "password": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
                "extra_body_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "file_path", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "OpenRouter"

    def generate(
        self,
        model,
        prompt,
        size,
        aspect_ratio,
        resolution,
        quality,
        output_format,
        seed,
        api_key,
        reference_image=None,
        provider_json="",
        extra_body_json="",
    ):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "output_format": output_format,
        }
        if size.strip():
            payload["size"] = size.strip()
        if aspect_ratio.strip() and aspect_ratio.strip() != "auto":
            payload["aspect_ratio"] = aspect_ratio.strip()
        if resolution.strip():
            payload["resolution"] = resolution.strip()
        if quality != "auto":
            payload["quality"] = quality
        if seed >= 0:
            payload["seed"] = seed
        if reference_image is not None:
            payload["input_references"] = _image_reference_payloads(reference_image, output_format)

        provider = _parse_json_object(provider_json, "provider_json")
        if provider:
            payload["provider"] = provider
        extra = _parse_json_object(extra_body_json, "extra_body_json")
        if extra:
            payload.update(extra)

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/images",
            headers=_headers(api_key),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_response(response)
        data = response.json()
        first = data.get("data", [{}])[0]
        b64_json = first.get("b64_json")
        if not b64_json:
            raise RuntimeError(f"OpenRouter image response did not include b64_json: {data}")

        media_type = first.get("media_type")
        ext = _safe_ext_from_mime(media_type, f".{output_format}")
        path = _unique_path("image", ext)
        raw = base64.b64decode(b64_json)
        path.write_bytes(raw)

        if media_type == "image/svg+xml" or ext == ".svg":
            raise RuntimeError(f"SVG image was saved to {path}, but ComfyUI IMAGE tensor output requires raster output.")

        return (_b64_image_to_tensor(b64_json), str(path), json.dumps(data, ensure_ascii=False, indent=2))


class OpenRouterVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_choices("video_model.txt"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": ("STRING", {"default": "720p"}),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60}),
                "poll_interval_seconds": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_wait_seconds": ("INT", {"default": 900, "min": 30, "max": 7200}),
                "api_key": ("STRING", {"default": "", "password": True}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
                "extra_body_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "job_id", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "OpenRouter"

    def generate(
        self,
        model,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        poll_interval_seconds,
        max_wait_seconds,
        api_key,
        first_frame_image=None,
        provider_json="",
        extra_body_json="",
    ):
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }
        if first_frame_image is not None:
            references = _image_reference_payloads(first_frame_image, "png")
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": references[0]["image_url"],
                    "frame_type": "first_frame",
                }
            ]
            if len(references) > 1:
                payload["input_references"] = references[1:]
        provider = _parse_json_object(provider_json, "provider_json")
        if provider:
            payload["provider"] = provider
        extra = _parse_json_object(extra_body_json, "extra_body_json")
        if extra:
            payload.update(extra)

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/videos",
            headers=_headers(api_key),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_response(response)
        job = response.json()
        job_id = job.get("id")
        polling_url = job.get("polling_url") or (f"{OPENROUTER_BASE_URL}/videos/{job_id}" if job_id else None)
        if not job_id or not polling_url:
            raise RuntimeError(f"OpenRouter video response did not include job id/polling_url: {job}")

        deadline = time.time() + max_wait_seconds
        current = job
        while time.time() < deadline:
            status = current.get("status")
            if status == "completed":
                urls = current.get("unsigned_urls") or []
                if not urls:
                    raise RuntimeError(f"Completed video job did not include unsigned_urls: {current}")
                download_url = urls[0]
                video_response = requests.get(download_url, headers=_headers(api_key), timeout=DEFAULT_TIMEOUT)
                _raise_for_response(video_response)
                content_type = video_response.headers.get("Content-Type", "")
                ext = _safe_ext_from_mime(content_type, ".mp4")
                path = _unique_path("video", ext)
                path.write_bytes(video_response.content)
                return (str(path), str(job_id), json.dumps(current, ensure_ascii=False, indent=2))
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"OpenRouter video job ended with status {status}: {current}")
            time.sleep(poll_interval_seconds)
            poll_response = requests.get(polling_url, headers=_headers(api_key), timeout=DEFAULT_TIMEOUT)
            _raise_for_response(poll_response)
            current = poll_response.json()

        raise RuntimeError(f"OpenRouter video job timed out after {max_wait_seconds}s: {current}")


class OpenRouterTTSAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_choices("audio_model.txt"),),
                "input_text": ("STRING", {"multiline": True, "default": ""}),
                "voice": ("STRING", {"default": "alloy"}),
                "response_format": (["mp3", "pcm"], {"default": "mp3"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "api_key": ("STRING", {"default": "", "password": True}),
            },
            "optional": {
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
                "extra_body_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_path", "generation_id")
    FUNCTION = "generate"
    CATEGORY = "OpenRouter"

    def generate(self, model, input_text, voice, response_format, speed, api_key, provider_json="", extra_body_json=""):
        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        provider = _parse_json_object(provider_json, "provider_json")
        if provider:
            payload["provider"] = provider
        extra = _parse_json_object(extra_body_json, "extra_body_json")
        if extra:
            payload.update(extra)

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/audio/speech",
            headers=_headers(api_key),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_response(response)
        ext = ".mp3" if response_format == "mp3" else ".pcm"
        path = _unique_path("audio", ext)
        path.write_bytes(response.content)
        return (str(path), response.headers.get("X-Generation-Id", ""))


NODE_CLASS_MAPPINGS = {
    "OpenRouterText": OpenRouterText,
    "OpenRouterImage": OpenRouterImage,
    "OpenRouterVideo": OpenRouterVideo,
    "OpenRouterTTSAudio": OpenRouterTTSAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterText": "OpenRouter Text",
    "OpenRouterImage": "OpenRouter Image",
    "OpenRouterVideo": "OpenRouter Video",
    "OpenRouterTTSAudio": "OpenRouter TTS Audio",
}
