"""Hermes plugin: voice-tone STT provider.

Registers a transcription provider named ``voice-tone`` that POSTs audio to
the voice-tone container (faster-whisper + emotion2vec) and returns the
transcript with an optional tone tag, e.g.::

    I'm done for today.
    [tone: neutral]

Config (config.yaml)::

  stt:
    provider: voice-tone

Env vars:
  VOICE_TONE_URL     container base URL (default http://localhost:8190)
  TONE_CONFIDENCE    min emotion confidence to emit a tag (default 0.6)
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any, Dict, Optional


def register(context) -> None:  # noqa: ANN001 - PluginContext
    context.register_transcription_provider(Provider())


class Provider:
    name = "voice-tone"

    def __init__(self) -> None:
        self.base_url = os.environ.get("VOICE_TONE_URL", "http://localhost:8190").rstrip("/")

    def default_model(self) -> str:
        return "small"

    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None, **extra: Any
    ) -> Dict[str, Any]:
        import json
        import mimetypes

        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        boundary = "----voice-tone-boundary-9f2c"
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as fh:
            audio = fh.read()

        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                audio,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )

        req = urllib.request.Request(
            f"{self.base_url}/transcribe",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - envelope contract: never raise
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"voice-tone container error: {exc}",
            }

        text = (payload.get("text") or "").strip()
        tone = payload.get("tone")
        threshold = float(os.environ.get("TONE_CONFIDENCE", "0.6"))
        if tone and float(payload.get("confidence", 0)) >= threshold:
            text = f"{text}\n[tone: {tone}]" if text else f"[tone: {tone}]"

        return {
            "success": bool(text),
            "transcript": text,
            "provider": self.name,
            **({"error": "empty transcript"} if not text else {}),
        }
