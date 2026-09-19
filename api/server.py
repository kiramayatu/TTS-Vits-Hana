"""FastAPI server + lightweight browser UI for Hana VITS."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.engine import HanaVITSEngine
from core.text import LANGUAGES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("HANA_MODEL", ROOT / "models" / "G_latest.pth"))
CONFIG_PATH = Path(os.getenv("HANA_CONFIG", ROOT / "config" / "fine.json"))
DEVICE = os.getenv("HANA_DEVICE", "auto")
HOST = os.getenv("HANA_HOST", "127.0.0.1")
PORT = int(os.getenv("HANA_PORT", "7860"))
MAX_TEXT_LENGTH = int(os.getenv("HANA_MAX_TEXT_LENGTH", "2000"))

engine = HanaVITSEngine(MODEL_PATH, CONFIG_PATH, DEVICE)

app = FastAPI(
    title="Hana VITS TTS",
    version="1.0.0-modern-stage1",
    description="Standalone FastAPI runtime for the preserved Hana VITS checkpoint.",
)

cors_origins = os.getenv("HANA_CORS_ORIGINS", "http://127.0.0.1:7860,http://localhost:7860")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in cors_origins.split(",") if item.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: str = "Hana"
    language: str = "日本語"
    speed: float = Field(default=1.0, gt=0.09, le=5.0)


@app.on_event("startup")
def startup() -> None:
    engine.load()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **engine.status()}


@app.get("/status")
def status() -> dict:
    return engine.status()


@app.get("/speakers")
def speakers() -> dict:
    return {"speakers": engine.speakers, "languages": LANGUAGES}


def synthesize_response(request: TTSRequest, download: bool) -> Response:
    try:
        result = engine.synthesize(request.text, request.speaker, request.language, request.speed)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
    payload = buffer.getvalue()

    safe_speaker = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.speaker)
    filename = f"hana-{safe_speaker}-{int(time.time() * 1000)}.wav"
    disposition = "attachment" if download else "inline"

    headers = {
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "X-TTS-Generation-Ms": f"{result.generation_ms:.2f}",
        "X-TTS-Audio-Seconds": f"{result.audio_seconds:.3f}",
        "X-TTS-RTF": f"{result.rtf:.4f}",
        "X-TTS-Device": result.device,
        "X-TTS-GPU-Memory-MB": f"{result.gpu_memory_mb:.2f}",
    }
    return Response(content=payload, media_type="audio/wav", headers=headers)


@app.post("/tts", response_class=Response)
def tts(request: TTSRequest, download: bool = Query(default=False)) -> Response:
    return synthesize_response(request, download)


@app.post("/v1/tts", response_class=Response)
def tts_v1(request: TTSRequest, download: bool = Query(default=False)) -> Response:
    return synthesize_response(request, download)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False, workers=1)
