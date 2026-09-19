# Hana VITS — Modern Stage 1

This is the first modernization stage of the old Hana VITS project.

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
