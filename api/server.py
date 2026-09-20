"""FastAPI server + lightweight browser UI for Hana VITS Stage 2."""
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.engine import HanaVITSEngine, TTSResult
from core.text import LANGUAGES
from core.sentence import SentenceSplitter, clean_tts_text, split_sentences

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("HANA_MODEL", ROOT / "models" / "G_latest.pth"))
CONFIG_PATH = Path(os.getenv("HANA_CONFIG", ROOT / "config" / "fine.json"))
DEVICE = os.getenv("HANA_DEVICE", "auto")
DTYPE = os.getenv("HANA_DTYPE", "fp32")
HOST = os.getenv("HANA_HOST", "127.0.0.1")
PORT = int(os.getenv("HANA_PORT", "7860"))
MAX_TEXT_LENGTH = int(os.getenv("HANA_MAX_TEXT_LENGTH", "2000"))
CACHE_ITEMS = int(os.getenv("HANA_CACHE_ITEMS", "32"))
CACHE_MAX_MB = int(os.getenv("HANA_CACHE_MAX_MB", "64"))
TEXT_CACHE_ITEMS = int(os.getenv("HANA_TEXT_CACHE_ITEMS", "128"))
REMOVE_WEIGHT_NORM = os.getenv("HANA_REMOVE_WEIGHT_NORM", "1").lower() not in {"0", "false", "no", "off"}
COMPILE_MODEL = os.getenv("HANA_TORCH_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
QUEUE_SIZE = max(1, int(os.getenv("HANA_QUEUE_SIZE", "4")))
MAX_SENTENCE_CHARS = max(16, int(os.getenv("HANA_SENTENCE_MAX_CHARS", "140")))
SEGMENT_GAP_MS = max(0, int(os.getenv("HANA_SEGMENT_GAP_MS", "30")))
STREAM_MAX_PENDING = max(1, int(os.getenv("HANA_STREAM_MAX_PENDING", "16")))
DEFAULT_NOISE_SCALE = float(os.getenv("HANA_NOISE_SCALE", "0.667"))
DEFAULT_NOISE_SCALE_W = float(os.getenv("HANA_NOISE_SCALE_W", "0.8"))
LOUDNESS_MODE = os.getenv("HANA_LOUDNESS_MODE", "peak").lower()
LOUDNESS_TARGET_DBFS = float(os.getenv("HANA_LOUDNESS_TARGET_DBFS", "-18.0"))

engine = HanaVITSEngine(
    MODEL_PATH,
    CONFIG_PATH,
    DEVICE,
    DTYPE,
    remove_weight_norm=REMOVE_WEIGHT_NORM,
    compile_model=COMPILE_MODEL,
    cache_items=CACHE_ITEMS,
    cache_max_mb=CACHE_MAX_MB,
    text_cache_items=TEXT_CACHE_ITEMS,
    noise_scale=DEFAULT_NOISE_SCALE,
    noise_scale_w=DEFAULT_NOISE_SCALE_W,
    loudness_mode=LOUDNESS_MODE,
    loudness_target_dbfs=LOUDNESS_TARGET_DBFS,
)


class InferenceQueue:
    """Single GPU inference worker with a bounded FIFO queue."""

    def __init__(self, maxsize: int) -> None:
        self._queue: queue.Queue[tuple[Future[TTSResult], Callable[..., TTSResult], tuple[object, ...]] | None] = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self._thread = threading.Thread(target=self._run, name="hana-tts-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            future, func, args = item
            try:
                if not future.cancelled():
                    future.set_result(func(*args))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _submit(self, func: Callable[..., TTSResult], *args: object) -> TTSResult:
        if not self._started:
            self.start()
        future: Future[TTSResult] = Future()
        try:
            self._queue.put_nowait((future, func, args))
        except queue.Full as exc:
            raise RuntimeError("TTS queue is full; retry shortly") from exc
        return future.result()

    def submit(self, *args: object) -> TTSResult:
        return self._submit(engine.synthesize, *args)

    def submit_grouped(self, *args: object) -> TTSResult:
        return self._submit(engine.synthesize_segmented, *args)

    def stats(self) -> dict[str, int | bool]:
        return {"running": self._started, "queued": self._queue.qsize(), "max_size": self.maxsize}


inference_queue = InferenceQueue(QUEUE_SIZE)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.load()
    inference_queue.start()
    yield


app = FastAPI(
    title="Hana VITS TTS",
    version="1.2.3-modern-stage2.3",
    description="Standalone optimized FastAPI runtime for the preserved Hana VITS checkpoint with shared sentence-based generation and streaming.",
    lifespan=lifespan,
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
    language: str = "Japanese"
    speed: float = Field(default=1.0, gt=0.09, le=5.0)
    noise_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    noise_scale_w: float | None = Field(default=None, ge=0.0, le=2.0)
    sentence_max: int = Field(default=MAX_SENTENCE_CHARS, ge=16, le=500)


class SplitRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    max_chars: int = Field(default=MAX_SENTENCE_CHARS, ge=16, le=500)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **engine.status(), "queue": inference_queue.stats()}


@app.get("/status")
def status() -> dict:
    return {**engine.status(), "queue": inference_queue.stats()}


@app.get("/speakers")
def speakers() -> dict:
    return {"speakers": engine.speakers, "languages": LANGUAGES}


def result_to_wav(result: TTSResult) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _response_headers(request: TTSRequest, result: TTSResult) -> dict[str, str]:
    safe_speaker = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.speaker)
    suffix = "-cached" if result.cache_hit else ""
    filename = f"hana-{safe_speaker}{suffix}-{int(time.time() * 1000)}.wav"
    return {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-TTS-Generation-Ms": f"{result.generation_ms:.2f}",
        "X-TTS-Audio-Seconds": f"{result.audio_seconds:.3f}",
        "X-TTS-RTF": f"{result.rtf:.4f}",
        "X-TTS-Device": result.device,
        "X-TTS-Dtype": result.dtype,
        "X-TTS-GPU-Memory-MB": f"{result.gpu_memory_mb:.2f}",
        "X-TTS-Cache-Hit": "1" if result.cache_hit else "0",
        "X-TTS-Text-Cache-Hit": "1" if result.text_cache_hit else "0",
        "X-TTS-Preprocess-Ms": f"{result.preprocess_ms:.2f}",
        "X-TTS-Inference-Ms": f"{result.inference_ms:.2f}",
        "X-TTS-Segments": str(result.segment_count),
        "X-TTS-Segmented": "1" if result.segmented else "0",
        "X-TTS-Segment-Gap-Ms": str(SEGMENT_GAP_MS),
    }


def _submit_request(request: TTSRequest) -> TTSResult:
    return inference_queue.submit(
        request.text,
        request.speaker,
        request.language,
        request.speed,
        request.noise_scale,
        request.noise_scale_w,
    )


def _submit_grouped_request(request: TTSRequest) -> TTSResult:
    return inference_queue.submit_grouped(
        request.text,
        request.speaker,
        request.language,
        request.speed,
        request.noise_scale,
        request.noise_scale_w,
        request.sentence_max,
        SEGMENT_GAP_MS,
    )


def synthesize_response(request: TTSRequest, download: bool) -> Response:
    try:
        result = _submit_grouped_request(request)
    except RuntimeError as exc:
        code = 429 if "queue is full" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = _response_headers(request, result)
    if download:
        headers["Content-Disposition"] = headers["Content-Disposition"].replace("inline", "attachment", 1)
    return Response(content=result_to_wav(result), media_type="audio/wav", headers=headers)


@app.post("/tts", response_class=Response)
def tts(request: TTSRequest, download: bool = Query(default=False)) -> Response:
    return synthesize_response(request, download)


@app.post("/v1/tts", response_class=Response)
def tts_v1(request: TTSRequest, download: bool = Query(default=False)) -> Response:
    return synthesize_response(request, download)


@app.post("/text/split")
def text_split(request: SplitRequest) -> dict[str, object]:
    cleaned = clean_tts_text(request.text)
    return {"cleaned_text": cleaned, "segments": split_sentences(request.text, request.max_chars)}


class _FlushSignal:
    def __init__(self, future: asyncio.Future[None], generation_id: int) -> None:
        self.future = future
        self.generation_id = generation_id


async def _send_stream_segment(
    websocket: WebSocket,
    segment_index: int,
    segment_text: str,
    settings_snapshot: dict[str, object],
    generation_id: int,
    current_generation: Callable[[], int],
) -> tuple[bool, bytes | None]:
    request = TTSRequest(
        text=segment_text,
        speaker=str(settings_snapshot["speaker"]),
        language=str(settings_snapshot["language"]),
        speed=float(settings_snapshot["speed"]),
        noise_scale=float(settings_snapshot["noise_scale"]),
        noise_scale_w=float(settings_snapshot["noise_scale_w"]),
    )
    try:
        result = await asyncio.to_thread(_submit_request, request)
        if generation_id != int(current_generation()):
            return False, None
        wav = result_to_wav(result)
        await websocket.send_json(
            {
                "type": "segment",
                "index": segment_index,
                "text": segment_text,
                "audio_seconds": result.audio_seconds,
                "generation_ms": result.generation_ms,
                "rtf": result.rtf,
                "cache_hit": result.cache_hit,
                "text_cache_hit": result.text_cache_hit,
                "segment_gap_ms": SEGMENT_GAP_MS,
            }
        )
        await websocket.send_bytes(wav)
        return True, wav
    except Exception as exc:
        # A browser disconnect is expected during interruption. Treat it the
        # same as cancellation and avoid leaking task exceptions.
        if generation_id == int(current_generation()):
            try:
                await websocket.send_json({"type": "error", "segment": segment_index, "detail": str(exc)})
            except Exception:
                pass
        return False, None


@app.websocket("/ws/tts")
async def tts_stream(websocket: WebSocket) -> None:
    """Incremental sentence-to-audio websocket.

    Client protocol:
      {"action":"start", "speaker":"Hana", "language":"Japanese", "speed":1.0}
      {"action":"append", "text":"こんにちは"}
      {"action":"flush"}
      {"action":"cancel"}
      {"action":"close"}

    The client may send multiple `append` messages while later text is still
    being generated. Complete sentences are placed on a per-connection queue;
    one sequential stream worker owns the generation order so no segment is
    lost when the global GPU queue is briefly full.
    """
    await websocket.accept()
    splitter = SentenceSplitter(MAX_SENTENCE_CHARS)
    segment_queue: asyncio.Queue[tuple[int, str, dict[str, object], int] | _FlushSignal] = asyncio.Queue(
        maxsize=STREAM_MAX_PENDING
    )
    settings: dict[str, object] = {
        "speaker": "Hana",
        "language": "Japanese",
        "speed": 1.0,
        "noise_scale": DEFAULT_NOISE_SCALE,
        "noise_scale_w": DEFAULT_NOISE_SCALE_W,
        "generation_id": 0,
    }
    next_index = 0

    async def stream_worker() -> None:
        try:
            while True:
                item = await segment_queue.get()
                try:
                    if isinstance(item, _FlushSignal):
                        if item.generation_id == int(settings["generation_id"]):
                            if not item.future.done():
                                item.future.set_result(None)
                            await websocket.send_json({"type": "done", "segments": next_index})
                        continue

                    index, text, snapshot, generation_id = item
                    await _send_stream_segment(
                        websocket,
                        index,
                        text,
                        snapshot,
                        generation_id,
                        lambda: int(settings["generation_id"]),
                    )
                finally:
                    segment_queue.task_done()
        except asyncio.CancelledError:
            raise
    worker_task = asyncio.create_task(stream_worker())

    def drain_segment_queue() -> None:
        while True:
            try:
                item = segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, _FlushSignal):
                if not item.future.done():
                    item.future.set_result(None)
            segment_queue.task_done()

    async def enqueue_segments(segments: list[str], generation_id: int) -> None:
        nonlocal next_index
        snapshot = dict(settings)
        for segment in segments:
            await segment_queue.put((next_index, segment, snapshot, generation_id))
            next_index += 1

    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            action = str(payload.get("action", "")).lower()

            if action == "start":
                drain_segment_queue()
                splitter = SentenceSplitter(int(payload.get("max_chars", MAX_SENTENCE_CHARS)))
                settings.update(
                    {
                        "speaker": payload.get("speaker", "Hana"),
                        "language": payload.get("language", "Japanese"),
                        "speed": float(payload.get("speed", 1.0)),
                        "noise_scale": DEFAULT_NOISE_SCALE if payload.get("noise_scale") is None else float(payload["noise_scale"]),
                        "noise_scale_w": DEFAULT_NOISE_SCALE_W if payload.get("noise_scale_w") is None else float(payload["noise_scale_w"]),
                        "generation_id": int(settings["generation_id"]) + 1,
                    }
                )
                next_index = 0
                await websocket.send_json({"type": "ready", "max_chars": splitter.max_chars, "segment_gap_ms": SEGMENT_GAP_MS})

            elif action == "append":
                chunk = str(payload.get("text", ""))
                if not chunk:
                    continue
                generation_id = int(settings["generation_id"])
                try:
                    await enqueue_segments(splitter.feed(chunk), generation_id)
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            elif action == "flush":
                generation_id = int(settings["generation_id"])
                try:
                    await enqueue_segments(splitter.flush(), generation_id)
                    loop = asyncio.get_running_loop()
                    done = loop.create_future()
                    await segment_queue.put(_FlushSignal(done, generation_id))
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            elif action == "cancel":
                settings["generation_id"] = int(settings["generation_id"]) + 1
                splitter.reset()
                drain_segment_queue()
                await websocket.send_json({"type": "cancelled"})

            elif action == "close":
                await websocket.close(code=1000)
                return
            else:
                await websocket.send_json({"type": "error", "detail": f"Unknown action: {action}"})
    except (WebSocketDisconnect, json.JSONDecodeError):
        pass
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False, workers=1)
