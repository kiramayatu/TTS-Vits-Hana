# Hana VITS — Modern Stage 2

This build contains the Stage 1 standalone runtime plus Stage 2 performance and reliability upgrades.

## What changed

- The original `G_latest.pth` is preserved byte-for-byte.
- The original Gradio application is moved to `legacy/app_legacy.py`.
- A standalone FastAPI server is added in `api/server.py`.
- A small HTML/CSS/JS web UI replaces Gradio for manual testing.
- The new UI exposes a single canonical `Japanese` option; legacy `日本語` API requests remain accepted as an alias.
- Generated audio is conservatively peak-normalized to avoid the very quiet output level of the legacy inference path.
- Generated WAV audio can be played in-browser and downloaded.
- `/tts` and `/v1/tts` are available for integration.
- `/health`, `/status`, `/speakers`, and FastAPI `/docs` are available.
- VITS model loading/checkpoint handling is isolated from HTTP/UI code.
- Inference uses `model.eval()` and `torch.inference_mode()`.
- The model is loaded once at startup and kept resident.
- A single inference lock prevents overlapping GPU requests in this stage.
- The API reports generation time, audio duration, RTF, device, and GPU memory.

## Preserved model

- Model: `models/G_latest.pth`
- Config: `config/fine.json`
- Hana speaker ID: `0`
- Sample rate: `22050 Hz`
- Original checkpoint SHA-256 is stored in `models/G_latest.pth.sha256`.

## Recommended environment

Use Python 3.12 on the target Linux/Windows machine.

PyTorch 2.14.0 provides current CUDA builds; the PyTorch project publishes CUDA 13.0 wheels, including Linux/Windows Python 3.12 wheels. The minimal runtime intentionally omits the old training/VC/ONNX/Gradio dependency stack.

The legacy multilingual Chinese frontend is optional; install `requirements-multilingual.txt` only when Chinese input is needed. Japanese/English requests no longer import the Chinese frontend.

Example Linux setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# For the original Chinese frontend as well:
# python -m pip install -r requirements-multilingual.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Start

From the project root:

```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 7860
```

Then open:

- Web UI: `http://127.0.0.1:7860/`
- API docs: `http://127.0.0.1:7860/docs`
- Health: `http://127.0.0.1:7860/health`

## API

### POST `/tts`

Request:

```json
{
  "text": "こんにちは、マスター。",
  "speaker": "Hana",
  "language": "Japanese",
  "speed": 1.0
}
```

The response is a WAV file. The browser UI uses the same endpoint and creates a local Download WAV action from the returned audio.

Optional query parameter:

```text
POST /tts?download=true
```

sets the response to attachment mode for direct downloads.

### Compatibility alias

`POST /v1/tts` accepts the same request body.

## Environment variables

- `HANA_MODEL` — model path
- `HANA_CONFIG` — config path
- `HANA_DEVICE` — `auto`, `cpu`, or `cuda:0`
- `HANA_HOST` — bind address
- `HANA_PORT` — bind port
- `HANA_MAX_TEXT_LENGTH` — maximum request text length
- `HANA_CORS_ORIGINS` — comma-separated allowed origins

## Stage 1 verification

The included static/API tests were run against the available CPU environment. The model loaded successfully and the FastAPI endpoint generated a valid WAV response. The test environment did not have an NVIDIA CUDA device, so RTX 5060 Ti performance has not been benchmarked yet.

The Japanese path requires `pyopenjtalk`; the build keeps that dependency explicit. The multilingual Chinese frontend is optional and isolated so Japanese/English requests do not import it.

## Important scope for Stage 1

This stage intentionally does **not** modify the model architecture, retrain Hana, enable emotion conditioning, or convert the checkpoint to ONNX/TensorRT. Those come only after the new runtime has been benchmarked against the legacy application.

## Stage 2 — runtime optimization

Stage 2 keeps the same `G_latest.pth` and adds optional performance features:

- NVIDIA CUDA FP16 autocast (`HANA_DTYPE=fp16`) while keeping model parameters in FP32 for safer legacy-VITS inference. `auto` selects FP16 on CUDA and FP32 on CPU.
- Generator weight-normalization removal at startup (`HANA_REMOVE_WEIGHT_NORM=1`, default). This is an inference-only optimization; the checkpoint file is unchanged.
- Bounded single-worker FIFO inference queue (`HANA_QUEUE_SIZE=4`) so concurrent requests do not start multiple GPU inference jobs.
- Byte-budgeted LRU audio cache (`HANA_CACHE_ITEMS=32`, `HANA_CACHE_MAX_MB=64`). Cached requests skip VITS generation.
- Optional `torch.compile` flag (`HANA_TORCH_COMPILE=1`) for benchmarking. It is disabled by default because dynamic text lengths can cause recompilation overhead.
- `scripts/benchmark.py` measures generation time, audio duration, RTF, and GPU memory so FP16/compile changes can be compared on the real RTX 5060 Ti.
- `/status` and `/health` now expose dtype, optimization state, cache state, and queue state.
- `POST /cache/clear` clears the generated-audio cache.

### Stage 2 environment variables

- `HANA_DTYPE=auto|fp16|fp32`
- `HANA_REMOVE_WEIGHT_NORM=1|0`
- `HANA_TORCH_COMPILE=1|0` — lazy `torch.compile` optimization; first uncached request includes compiler warm-up.
- `HANA_QUEUE_SIZE=4`
- `HANA_CACHE_ITEMS=32`
- `HANA_CACHE_MAX_MB=64`

For the first real benchmark on the RTX 5060 Ti, leave `HANA_TORCH_COMPILE=0` and compare `HANA_DTYPE=fp32` against `HANA_DTYPE=fp16`.

## Stage 2.2 — sentence streaming and voice controls

Stage 2.2 focuses on time-to-first-audio and avoiding very long single-shot VITS requests. The original `/tts` endpoint remains backward compatible and still returns one WAV response.

### Sentence segmentation

`core/sentence.py` provides:

- Japanese-safe `。！？` sentence boundaries.
- English `.` boundaries when they are not decimal numbers.
- Preservation of closing quotes/brackets after sentence punctuation.
- Long-fragment splitting at Japanese commas/whitespace before a hard cut.
- Common LLM formatting cleanup (role prefixes, markdown links/fences, repeated punctuation, bullet markers).
- Incremental `SentenceSplitter.feed()` / `flush()` for LLM token/chunk streams.

The default maximum sentence length is 140 characters and can be changed with `HANA_SENTENCE_MAX_CHARS`.

### WebSocket streaming

Endpoint:

```text
WS /ws/tts
```

Protocol:

```json
{"action":"start","speaker":"Hana","language":"Japanese","speed":1.0}
{"action":"append","text":"こんにちは、マスター。"}
{"action":"append","text":"今日はどうしますか？"}
{"action":"flush"}
```

The server emits a JSON `segment` message followed by the binary PCM16 WAV for each completed sentence. `cancel` invalidates stale results that are still finishing on the single GPU worker.

The WebSocket endpoint remains available as an integration path, while the browser UI uses it internally for progressive playback during the normal Generate Voice action.

### Voice controls

`POST /tts`, `/v1/tts`, and WebSocket start requests support:

```json
{
  "text": "こんにちは。",
  "speaker": "Hana",
  "language": "Japanese",
  "speed": 1.0,
  "noise_scale": 0.667,
  "noise_scale_w": 0.8
}
```

The defaults remain the existing VITS settings. The lightweight web UI exposes them under **Advanced voice controls** so you can test the breathy/sighing behavior without changing the checkpoint.

### Streaming playback

The browser UI's **Stream Sentences** button uses WebSocket sentence streaming and schedules each WAV segment through the Web Audio API. This is a TTS-side latency improvement; the Open-LLM-VTuber adapter has not been changed in Stage 2.2 yet.

The browser can also combine the received PCM16 WAV segments into one downloadable WAV without rerunning TTS.

### FP32 policy

The production runtime explicitly defaults to:

```text
HANA_DTYPE=fp32
HANA_TORCH_COMPILE=0
```

This follows the target-machine benchmark where FP32 outperformed the tested FP16 and FP16+compile configurations for the legacy Hana VITS workload.

### Loudness handling

The default `HANA_LOUDNESS_MODE=peak` preserves the Stage 1.1 behavior: generated audio is peak-normalized toward -1 dBFS with a conservative gain cap. An optional `rms` mode can target `HANA_LOUDNESS_TARGET_DBFS` (default -18 dBFS) while still respecting the -1 dBFS peak ceiling. `off` disables gain normalization. The default remains `peak` so Stage 2.2 does not silently change Hana's established timbre/level behavior.

## Shared sentence generation (Stage 2.3)

`POST /tts` and `POST /v1/tts` now use the same sentence-sized generation units as the WebSocket streamer. Generate Voice uses sentence-sized units internally, inserts the configured inter-segment gap, starts playback as each unit becomes ready, and combines those exact units into one final WAV. There is no separate Stream Sentences button in the web UI; progressive playback is now part of the normal Generate Voice flow.

The sentence size is controlled by `sentence_max` in the JSON API or the **Sentence max chars** control in the web UI. The server-wide inter-segment gap is `HANA_SEGMENT_GAP_MS` (default `30` ms).
