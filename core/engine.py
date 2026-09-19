"""Thread-safe VITS inference engine for the preserved Hana checkpoint."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.hparams import HParams, load_checkpoint, load_hparams
from core.text import LANGUAGE_MARKS, LANGUAGES, prepare_text
from models import SynthesizerTrn


@dataclass(frozen=True)
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    generation_ms: float
    audio_seconds: float
    rtf: float
    device: str
    gpu_memory_mb: float


class HanaVITSEngine:
    def __init__(self, model_path: Path, config_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.requested_device = device
        self.lock = threading.Lock()
        self.model: torch.nn.Module | None = None
        self.hps: HParams | None = None
        self.speakers: dict[str, int] = {}
        self.device = self._resolve_device(device)
        self.iteration = 0
        self.learning_rate = 0.0

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device(requested)

    def load(self) -> None:
        hps = load_hparams(self.config_path)
        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        model, iteration, learning_rate = load_checkpoint(self.model_path, model)
        model = model.to(self.device)
        model.eval()

        self.hps = hps
        self.model = model
        self.speakers = dict(hps.speakers)
        self.iteration = iteration
        self.learning_rate = learning_rate

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.hps is not None

    def status(self) -> dict[str, Any]:
        gpu = None
        if self.device.type == "cuda":
            idx = self.device.index or torch.cuda.current_device()
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 2),
                "memory_total_mb": round(torch.cuda.get_device_properties(idx).total_memory / 1024**2, 2),
            }
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "speakers": self.speakers,
            "sample_rate": self.hps.data.sampling_rate if self.hps else None,
            "checkpoint_iteration": self.iteration,
            "gpu": gpu,
        }

    def synthesize(self, text: str, speaker: str, language: str, speed: float) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("TTS model is not loaded")
        if speaker not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker}")
        if language not in LANGUAGE_MARKS:
            raise ValueError(f"Unsupported language: {language}")
        if speed <= 0:
            raise ValueError("speed must be greater than 0")

        assert self.hps is not None
        assert self.model is not None

        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()

            stn = prepare_text(text, language, self.hps)
            x = stn.unsqueeze(0).to(self.device, non_blocking=True)
            x_lengths = torch.tensor([stn.numel()], dtype=torch.long, device=self.device)
            sid = torch.tensor([self.speakers[speaker]], dtype=torch.long, device=self.device)

            with torch.inference_mode():
                output = self.model.infer(
                    x,
                    x_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0 / speed,
                )[0][0, 0]

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            audio = output.detach().float().cpu().numpy()
            elapsed = time.perf_counter() - started

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # The legacy VITS output can be noticeably quieter than normal WAV
        # speech levels. Normalize peaks conservatively so the browser and
        # Open-LLM-VTuber receive a usable level without hard-clipping.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            target_peak = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
            max_gain = 10.0 ** (24.0 / 20.0)      # cap at +24 dB
            gain = min(target_peak / peak, max_gain)
            audio = audio * gain

        audio = np.clip(audio, -1.0, 1.0)
        sample_rate = int(self.hps.data.sampling_rate)
        audio_seconds = len(audio) / sample_rate
        gpu_mb = 0.0
        if self.device.type == "cuda":
            gpu_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            generation_ms=elapsed * 1000.0,
            audio_seconds=audio_seconds,
            rtf=elapsed / audio_seconds if audio_seconds else 0.0,
            device=str(self.device),
            gpu_memory_mb=gpu_mb,
        )
