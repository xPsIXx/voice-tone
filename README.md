# voice-tone

Whisper transcription + speech emotion tagging in one small CPU container, with a
Hermes STT provider plugin that returns tone-tagged transcripts.

```
voice note ──> voice-tone container (faster-whisper + emotion2vec+)
                    │  {text, tone, confidence}
                    ▼
             Hermes plugin "voice-tone"
                    │  transcript + "[tone: neutral]"
                    ▼
                 agent sees it
```

## Components

- **Container** (`app/main.py`, `Dockerfile`): FastAPI service.
  - `GET /health` — liveness + loaded model info
  - `POST /transcribe` — multipart audio upload → `{text, tone, confidence, language}`
  - Whisper via faster-whisper (int8, CPU); emotion via emotion2vec+ base (9 classes)
  - Tone is only emitted when confidence ≥ threshold, otherwise `tone: null`
- **Hermes plugin** (`__init__.py`, `plugin.yaml`): registers a transcription
  provider named `voice-tone`. No pip dependencies — stdlib only.

## Environment variables

| Var | Default | Where | Meaning |
|-----|---------|-------|---------|
| `WHISPER_MODEL` | `small` | container | faster-whisper model name |
| `EMOTION_MODEL` | `emotion2vec/emotion2vec_plus_base` | container | HF emotion model |
| `TONE_CONFIDENCE` | `0.6` | both | min confidence to emit a tone tag |
| `WHISPER_LANGUAGE` | auto | container | force BCP-47 language |
| `VOICE_TONE_URL` | `http://localhost:8190` | plugin | container base URL |

No secrets required. Models download from HuggingFace on first start into the
`/models` volume — mount a persistent path there so updates don't re-download.

## Run

```bash
docker run -d --name voice-tone \
  -p 8190:8190 \
  -v voice-tone-models:/models \
  ghcr.io/<owner>/voice-tone:latest
```

First request is slow (model load ~30-60s); check `GET /health` until ready.

## Hermes setup

1. Copy the plugin files into your plugins dir:
   ```bash
   mkdir -p ~/.hermes/plugins/voice-tone
   cp __init__.py plugin.yaml ~/.hermes/plugins/voice-tone/
   ```
2. `config.yaml`:
   ```yaml
   stt:
     provider: voice-tone
   ```
3. Restart the gateway. Rollback = set `stt.provider` back to your previous value.

## CI/CD

GitHub Actions builds and pushes to GHCR on push to `main` (tagged `latest`)
and on `v*` tags (semver + sha tags). Uses `GITHUB_TOKEN`, no PAT needed.

## Notes / known limitations

- Emotion accuracy is benchmark-grade, not mind-reading: IEMOCAP-style numbers
  (73-81%) are on *acted* speech; casual talking scores lower. A confidence
  threshold keeps low-confidence tags out of the transcript rather than guessing.
- CPU only by design — no GPU dependency, no VRAM contention.
- Optional Stage 2 (not in this repo): a Qwen2.5-Omni pass for free-form tone
  descriptions beyond the 9 fixed classes.
