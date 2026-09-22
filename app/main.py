"""voice-tone: Whisper transcription + emotion2vec tone tagging.

Single service that returns a transcript with an optional emotion tag,
designed to back Hermes' STT provider plugin.

Endpoints:
  GET  /health      -> liveness + loaded model info
  POST /transcribe  -> multipart audio upload -> {text, tone, confidence}

Models load during startup (lifespan); the server only accepts traffic once
both models are ready.
"""

import io
import os
import tempfile
import time
import wave

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from funasr import AutoModel

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
# ModelScope id by default (guaranteed path per FunASR docs). If ModelScope is
# unreachable from your network, set EMOTION_MODEL=emotion2vec/emotion2vec_plus_base HUB=hf
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "iic/emotion2vec_plus_base")
HUB = os.environ.get("HUB", "ms")  # "ms" (ModelScope) or "hf" (HuggingFace)
TONE_CONFIDENCE = float(os.environ.get("TONE_CONFIDENCE", "0.6"))
SAMPLE_RATE = 16000

whisper: WhisperModel | None = None
emo_model: AutoModel | None = None


async def lifespan(_: FastAPI):
    """Load models before the server accepts traffic."""
    global whisper, emo_model
    t0 = time.time()
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    emo_model = AutoModel(model=EMOTION_MODEL, hub=HUB, device="cpu", disable_update=True)
    print(f"models ready in {time.time() - t0:.1f}s", flush=True)
    yield


app = FastAPI(title="voice-tone", lifespan=lifespan)


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

    # emotion2vec+ via FunASR: res[0] -> {"labels": [...], "scores": [...]}
    tone, conf = None, 0.0
    try:
        clip = audio[: SAMPLE_RATE * 30]  # first 30s is plenty for utterance-level emotion
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            pcm = np.clip(clip * 32767, -32768, 32767).astype("<i2").tobytes()
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm)
            res = emo_model.generate(
                input=tmp.name, granularity="utterance", extract_embedding=False
            )
        if res and res[0].get("labels"):
            labels, scores = res[0]["labels"], res[0]["scores"]
            conf = max(float(s) for s in scores)
            best = labels[int(np.argmax(scores))]
            tone = str(best).split("/")[-1]  # strip any path prefix
            if conf < TONE_CONFIDENCE:
                tone = None  # low confidence -> no tag, plain transcript
    except Exception as exc:  # noqa: BLE001 - emotion is best-effort
        print(f"emotion pass failed: {exc}", flush=True)

    return {
        "text": text,
        "tone": tone,
        "confidence": round(conf, 3),
        "language": info.language,
    }
