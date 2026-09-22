"""voice-tone: Whisper transcription + emotion2vec tone tagging.

Single service that returns a transcript with an optional emotion tag,
designed to back Hermes' STT provider plugin.

Endpoints:
  GET  /health      -> liveness + loaded model info
  POST /transcribe  -> multipart audio upload -> {text, tone, confidence}
"""

import io
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from transformers import AutoModelForAudioClassification, AutoProcessor

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "emotion2vec/emotion2vec_plus_base")
TONE_CONFIDENCE = float(os.environ.get("TONE_CONFIDENCE", "0.6"))
SAMPLE_RATE = 16000

app = FastAPI(title="voice-tone")

whisper: WhisperModel | None = None
emo_model: AutoModelForAudioClassification | None = None
emo_proc: AutoProcessor | None = None


@app.on_event("startup")
def load_models() -> None:
    global whisper, emo_model, emo_proc
    t0 = time.time()
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    emo_model = AutoModelForAudioClassification.from_pretrained(EMOTION_MODEL)
    emo_model.eval()
    emo_proc = AutoProcessor.from_pretrained(EMOTION_MODEL)
    print(f"models ready in {time.time() - t0:.1f}s", flush=True)


def decode_audio(data: bytes) -> np.ndarray:
    from faster_whisper.audio import decode_audio as _decode

    return _decode(io.BytesIO(data), sampling_rate=SAMPLE_RATE)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "emotion_model": EMOTION_MODEL,
        "tone_confidence_threshold": TONE_CONFIDENCE,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    if whisper is None or emo_model is None:
        raise HTTPException(503, "models not loaded yet")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio upload")

    try:
        audio = decode_audio(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(415, f"could not decode audio: {exc}") from exc

    segments, info = whisper.transcribe(
        audio,
        language=os.environ.get("WHISPER_LANGUAGE") or None,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()

    import torch  # local import keeps startup lighter

    inputs = emo_proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = emo_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    conf, idx = probs.max(dim=0)
    id2label = emo_model.config.id2label
    tone = id2label[int(idx)] if float(conf) >= TONE_CONFIDENCE else None

    return {
        "text": text,
        "tone": tone,
        "confidence": round(float(conf), 3),
        "language": info.language,
    }
