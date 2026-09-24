
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

