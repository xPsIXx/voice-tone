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
  VOICE_TONE_URL         container base URL override (default: docker alias, then bridge IP)
  TONE_CONFIDENCE        min emotion confidence to emit a tag (default 0.6)
  VOICE_TONE_LOCAL_FALLBACK  fall back to in-process local whisper if container is down (default 1)
  VOICE_TONE_JOURNAL     Obsidian file for the per-note tone journal (default
                         /obsidian/Kyle/Voice-Tone-Journal.md, "0" disables)
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
            container_error = f"voice-tone container error: {exc}"
            fb = self._local_fallback(file_path)
            if fb is not None:
                return fb
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": container_error,
            }

        raw = (payload.get("text") or "").strip()
        tone = payload.get("tone")
        confidence = float(payload.get("confidence", 0))
        threshold = float(os.environ.get("TONE_CONFIDENCE", "0.6"))

        self._append_journal(raw, tone, confidence)

        # Tag only when there is actual speech: emotion2vec can hallucinate a
        # confident label on silence, which must not surface as a transcript.
        if raw and tone and confidence >= threshold:
            raw = f"{raw}\n[tone: {tone}]"

        return {
            "success": bool(raw),
            "transcript": raw,
            "provider": self.name,
            **({} if raw else {"error": "empty transcript"}),
        }

    def _journal_path(self) -> Optional[str]:
        """Obsidian tone-journal path; '0' disables logging entirely."""
        val = os.environ.get("VOICE_TONE_JOURNAL", "").strip()
        if val == "0":
            return None
        return val or "/obsidian/Kyle/Voice-Tone-Journal.md"

    def _append_journal(self, text: str, tone: Optional[str], confidence: float) -> None:
        """Append one row per note to the Obsidian tone journal. Best-effort only —
        a missing vault must never break transcription."""
        try:
            path = self._journal_path()
            if not path or not text:
                return
            from datetime import datetime

            now = datetime.now().astimezone()
            excerpt = " ".join(text.split())[:80].replace("|", "\\|")
            tone_cell = f"{tone} ({confidence:.2f})" if tone else f"— ({confidence:.2f})"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"| {now:%H:%M} | {tone_cell} | {excerpt} |\n")
        except Exception:  # noqa: BLE001 - journal is best-effort, never raise
            pass

    def _local_fallback(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Last-resort in-process transcription via Hermes' built-in local faster-whisper
        (same model singleton/cache as the 'local' provider — no emotion tag). Used only
        when the container is unreachable or errors. Returns None only when the fallback
        backend itself is unavailable (then the caller surfaces the original error)."""
        if os.environ.get("VOICE_TONE_LOCAL_FALLBACK", "1") != "1":
            return None
        try:
            from tools.transcription_tools import transcribe_audio_local_fallback
        except Exception:  # noqa: BLE001 - standalone QA / minimal env
            return None
        try:
            result = transcribe_audio_local_fallback(file_path, model="small")
        except Exception as exc:  # noqa: BLE001 - envelope contract: never raise
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"voice-tone container error (local fallback also failed): {exc}",
            }
        if not isinstance(result, dict) or not result.get("success"):
            # Local backend reported its own failure — surface it rather than the
            # generic connection error so the real cause is visible.
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"voice-tone container error (local fallback also failed): {result.get('error') if isinstance(result, dict) else 'unknown'}",
            }
        text = (result.get("transcript") or "").strip()
        return {
            "success": bool(text),
            "transcript": text,
            "provider": self.name,
            **({} if text else {"error": "empty transcript"}),
        }
