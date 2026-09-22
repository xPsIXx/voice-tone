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

try:  # inside Hermes: must subclass the ABC or the registry rejects us (isinstance check)
    from agent.transcription_provider import TranscriptionProvider as _Base
except ImportError:  # standalone QA harness outside Hermes
    class _Base:  # type: ignore[no-redef]
        pass


def register(context) -> None:  # noqa: ANN001 - PluginContext
    context.register_transcription_provider(Provider())


class Provider(_Base):
    name = "voice-tone"

    def _base_urls(self) -> list[str]:
        """Candidate container URLs, tried in order per call (not cached) so .env
        edits apply without a gateway restart. Order: explicit env override,
        docker network alias, host bridge IP (port 8192). Set
        VOICE_TONE_SINGLE_URL=1 to try only the explicit URL (QA isolation)."""
        candidates = []
        url = os.environ.get("VOICE_TONE_URL", "").rstrip("/")
        if url:
            candidates.append(url)
        elif os.environ.get("VOICE_TONE_SINGLE_URL") == "1":
            return []  # single mode without explicit URL -> error path (QA isolation)
        else:
            candidates += ["http://voice-tone:8190", "http://172.17.0.1:8192"]
        return candidates

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

        try:
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

            payload = None
            last_exc: Optional[Exception] = None
            for base in self._base_urls():
                try:
                    req = urllib.request.Request(
                        f"{base}/transcribe",
                        data=body,
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        payload = json.loads(resp.read().decode())
                    break
                except Exception as exc:  # noqa: BLE001 - try next candidate
                    last_exc = exc
            if payload is None:
                raise last_exc or RuntimeError("no container URL configured")
        except Exception as exc:  # noqa: BLE001 - envelope contract: never raise
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"voice-tone container error: {exc}",
            }

        raw = (payload.get("text") or "").strip()
        tone = payload.get("tone")
        threshold = float(os.environ.get("TONE_CONFIDENCE", "0.6"))
        # Tag only when there is actual speech: emotion2vec can hallucinate a
        # confident label on silence, which must not surface as a transcript.
        if raw and tone and float(payload.get("confidence", 0)) >= threshold:
            raw = f"{raw}\n[tone: {tone}]"

        return {
            "success": bool(raw),
            "transcript": raw,
            "provider": self.name,
            **({} if raw else {"error": "empty transcript"}),
        }
