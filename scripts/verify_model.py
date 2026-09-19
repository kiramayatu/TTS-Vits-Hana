from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.engine import HanaVITSEngine

model = ROOT / "models" / "G_latest.pth"
config = ROOT / "config" / "fine.json"

sha = hashlib.sha256(model.read_bytes()).hexdigest()
print(f"Checkpoint SHA256: {sha}")
engine = HanaVITSEngine(model, config, "cpu")
engine.load()
print(json.dumps(engine.status(), indent=2))
result = engine.synthesize("hello, this is a Hana VITS smoke test.", "Hana", "English", 1.0)
print({
    "sample_rate": result.sample_rate,
    "audio_seconds": result.audio_seconds,
    "generation_ms": round(result.generation_ms, 2),
    "rtf": round(result.rtf, 4),
    "device": result.device,
})
