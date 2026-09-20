"""VITS inference engine with optional GPU optimizations, caching, and metrics."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.hparams import HParams, load_checkpoint, load_hparams
from core.text import LANGUAGE_MARKS, LANGUAGES, prepare_text
from core.sentence import split_sentences
from models import SynthesizerTrn


@dataclass(frozen=True)
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    generation_ms: float
    audio_seconds: float
    rtf: float
    device: str
    dtype: str
    gpu_memory_mb: float
    cache_hit: bool = False
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    text_cache_hit: bool = False
    segment_count: int = 1
    segmented: bool = False


class AudioCache:
    """Small byte-budgeted LRU cache for generated PCM arrays."""

    def __init__(self, max_items: int = 32, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_items = max(0, int(max_items))
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[str, tuple[TTSResult, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_items > 0 and self.max_bytes > 0

    def get(self, key: str) -> TTSResult | None:
        if not self.enabled:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            result, size = item
            self._items.move_to_end(key)
            # Return an independent read-only copy so callers cannot mutate the cache.
            return TTSResult(
                audio=result.audio.copy(),
                sample_rate=result.sample_rate,
                generation_ms=0.0,
                audio_seconds=result.audio_seconds,
                rtf=0.0,
                device=result.device,
                dtype=result.dtype,
                gpu_memory_mb=result.gpu_memory_mb,
                cache_hit=True,
                preprocess_ms=0.0,
                inference_ms=0.0,
                text_cache_hit=False,
                segment_count=result.segment_count,
                segmented=result.segmented,
            )

    def put(self, key: str, result: TTSResult) -> None:
        if not self.enabled:
            return
        audio = np.asarray(result.audio, dtype=np.float32).copy()
        size = int(audio.nbytes)
        if size > self.max_bytes:
            return
        cached = TTSResult(
            audio=audio,
            sample_rate=result.sample_rate,
            generation_ms=result.generation_ms,
            audio_seconds=result.audio_seconds,
            rtf=result.rtf,
            device=result.device,
            dtype=result.dtype,
            gpu_memory_mb=result.gpu_memory_mb,
            cache_hit=False,
            preprocess_ms=result.preprocess_ms,
            inference_ms=result.inference_ms,
            text_cache_hit=result.text_cache_hit,
            segment_count=result.segment_count,
            segmented=result.segmented,
        )
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            while self._items and (len(self._items) >= self.max_items or self._bytes + size > self.max_bytes):
                _, (_, old_size) = self._items.popitem(last=False)
                self._bytes -= old_size
            self._items[key] = (cached, size)
            self._bytes += size

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "items": len(self._items),
                "bytes": self._bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


class TextSequenceCache:
    """LRU cache for CPU-side text/phoneme preprocessing results.

    Japanese G2P can be considerably more CPU-expensive than the small VITS
    inference itself. Keeping the token sequence on CPU lets repeated phrases
    (greetings, alerts, UI phrases) skip that work entirely.
    """

    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_items > 0

    def get(self, key: str) -> torch.Tensor | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items.pop(key, None)
            while len(self._items) >= self.max_items:
                self._items.popitem(last=False)
            self._items[key] = value

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "items": len(self._items),
                "max_items": self.max_items,
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class HanaVITSEngine:
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        device: str = "auto",
        dtype: str = "auto",
        remove_weight_norm: bool = True,
        compile_model: bool = False,
        cache_items: int = 32,
        cache_max_mb: int = 64,
        text_cache_items: int = 128,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        loudness_mode: str = "peak",
        loudness_target_dbfs: float = -18.0,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.requested_device = device
        self.requested_dtype = dtype
        self.remove_weight_norm_enabled = bool(remove_weight_norm)
        self.compile_requested = bool(compile_model)
        self._compiled_infer = None
        self.compile_error: str | None = None
        self.model: torch.nn.Module | None = None
        self.hps: HParams | None = None
        self.speakers: dict[str, int] = {}
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.iteration = 0
        self.learning_rate = 0.0
        self.cache = AudioCache(cache_items, cache_max_mb * 1024 * 1024)
        self.text_cache = TextSequenceCache(text_cache_items)
        self.noise_scale = self._validate_noise("noise_scale", noise_scale)
        self.noise_scale_w = self._validate_noise("noise_scale_w", noise_scale_w)
        self.loudness_mode = str(loudness_mode).lower()
        if self.loudness_mode not in {"peak", "rms", "off"}:
            raise ValueError("loudness_mode must be one of: peak, rms, off")
        self.loudness_target_dbfs = float(loudness_target_dbfs)
        if not -40.0 <= self.loudness_target_dbfs <= -3.0:
            raise ValueError("loudness_target_dbfs must be between -40.0 and -3.0")
        self._load_lock = threading.Lock()
        self._compile_warm = False
        self.compile_active = False
        self.weight_norm_removed = False

    @staticmethod
    def _validate_noise(name: str, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"{name} must be between 0.0 and 2.0")
        return value

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device(requested)

    def _resolve_dtype(self, requested: str) -> torch.dtype:
        value = requested.lower()
        if value == "auto":
            # Benchmarks on the target RTX 5060 Ti show this legacy VITS
            # model is faster in FP32 than autocast FP16. Keep FP32 the safe
            # default and expose FP16 only as an explicit opt-in.
            return torch.float32
        if value == "fp16":
            if self.device.type != "cuda":
                raise RuntimeError("FP16 mode is only enabled for CUDA in this build")
            return torch.float16
        if value == "fp32":
            return torch.float32
        raise ValueError("HANA_DTYPE must be one of: auto, fp16, fp32")

    def load(self) -> None:
        with self._load_lock:
            if self.is_loaded:
                return
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

            # Weight normalization is a training-time reparameterization. For
            # inference it can be removed from the generator without changing
            # the learned effective weights, reducing per-sample overhead.
            if self.remove_weight_norm_enabled and hasattr(model, "dec") and hasattr(model.dec, "remove_weight_norm"):
                try:
                    model.dec.remove_weight_norm()
                    self.weight_norm_removed = True
                except (RuntimeError, ValueError):
                    # Keep the model usable if a future checkpoint/module layout
                    # does not support removing every weight norm.
                    self.weight_norm_removed = False

            if self.dtype == torch.float16 and self.device.type == "cuda":
                # Keep parameters in FP32 and use autocast for safer mixed
                # precision. Some legacy VITS ops are less robust if every
                # parameter is permanently converted to FP16.
                model = model.to(dtype=torch.float32)

            self.hps = hps
            self.model = model
            self.speakers = dict(hps.speakers)
            self.iteration = iteration
            self.learning_rate = learning_rate

            # torch.compile is intentionally opt-in. Dynamic text lengths may
            # trigger recompilations, and compilation can be slower for short
            # one-off requests. We expose it for measured benchmarking.
            if self.compile_requested and hasattr(torch, "compile"):
                # Lazy compilation keeps startup fast. The first uncached
                # synthesis pays the compiler warm-up cost; benchmarks should
                # perform warm-up requests before measuring steady state.
                self._compiled_infer = None
                self.compile_active = False

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
            "dtype": str(self.dtype).replace("torch.", ""),
            "speakers": self.speakers,
            "sample_rate": self.hps.data.sampling_rate if self.hps else None,
            "checkpoint_iteration": self.iteration,
            "gpu": gpu,
            "optimizations": {
                "weight_norm_removed": self.weight_norm_removed,
                "torch_compile_requested": self.compile_requested,
                "torch_compile_active": self.compile_active,
                "torch_compile_error": self.compile_error,
                "noise_scale": self.noise_scale,
                "noise_scale_w": self.noise_scale_w,
                "loudness_mode": self.loudness_mode,
                "loudness_target_dbfs": self.loudness_target_dbfs,
            },
            "cache": self.cache.stats(),
            "text_cache": self.text_cache.stats(),
        }

    @staticmethod
    def _cache_key(text: str, speaker: str, language: str, speed: float, noise_scale: float, noise_scale_w: float) -> str:
        raw = (
            f"v4\0{speaker}\0{language}\0{speed:.6f}\0"
            f"{noise_scale:.6f}\0{noise_scale_w:.6f}\0{text}"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_infer_callable(self):
        if not self.compile_requested:
            return None
        if self.compile_error is not None:
            return None
        if self._compiled_infer is None:
            if not hasattr(torch, "compile"):
                self.compile_error = "torch.compile is unavailable in this PyTorch build"
                return None
            assert self.model is not None
            model = self.model

            def infer_impl(x, x_lengths, sid, length_scale, noise_scale, noise_scale_w):
                return model.infer(
                    x,
                    x_lengths,
                    sid=sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )[0][0, 0]

            try:
                self._compiled_infer = torch.compile(
                    infer_impl,
                    mode="reduce-overhead",
                    dynamic=True,
                )
            except Exception as exc:
                self.compile_error = f"compile setup failed: {type(exc).__name__}: {exc}"
                return None
            return self._compiled_infer
        return self._compiled_infer

    @staticmethod
    def _text_cache_key(text: str, language: str) -> str:
        raw = f"text-v1\0{language}\0{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _prepare_text_cached(self, text: str, language: str) -> tuple[torch.Tensor, bool]:
        assert self.hps is not None
        key = self._text_cache_key(text, language)
        cached = self.text_cache.get(key)
        if cached is not None:
            return cached, True
        sequence = prepare_text(text, language, self.hps)
        # Store a CPU tensor; the caller creates a GPU view/copy as needed.
        sequence = sequence.contiguous()
        self.text_cache.put(key, sequence)
        return sequence, False

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 1e-8 or self.loudness_mode == "off":
            return np.clip(audio, -1.0, 1.0)

        target_peak = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
        max_gain = 10.0 ** (24.0 / 20.0)      # cap at +24 dB
        if self.loudness_mode == "peak":
            gain = min(target_peak / peak, max_gain)
        else:
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            target_rms = 10.0 ** (self.loudness_target_dbfs / 20.0)
            rms_gain = target_rms / max(rms, 1e-8)
            peak_gain = target_peak / peak
            gain = min(rms_gain, peak_gain, 10.0 ** (12.0 / 20.0))
        return np.clip(audio * gain, -1.0, 1.0)

    @staticmethod
    def combine_audio_segments(segments: list[np.ndarray], sample_rate: int, gap_ms: int = 30) -> np.ndarray:
        """Concatenate PCM segments with a fixed inter-segment gap."""
        if not segments:
            return np.zeros(0, dtype=np.float32)
        normalized = [np.asarray(segment, dtype=np.float32).reshape(-1) for segment in segments]
        gap_samples = max(0, int(round(sample_rate * max(0, gap_ms) / 1000.0)))
        if gap_samples == 0 or len(normalized) == 1:
            return np.concatenate(normalized).astype(np.float32, copy=False)
        silence = np.zeros(gap_samples, dtype=np.float32)
        pieces: list[np.ndarray] = []
        for index, segment in enumerate(normalized):
            pieces.append(segment)
            if index + 1 < len(normalized):
                pieces.append(silence)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    @staticmethod
    def _segmented_cache_key(
        text: str,
        speaker: str,
        language: str,
        speed: float,
        noise_scale: float,
        noise_scale_w: float,
        max_chars: int,
        gap_ms: int,
    ) -> str:
        raw = (
            f"seg-v1\\0{speaker}\\0{language}\\0{speed:.6f}\\0"
            f"{noise_scale:.6f}\\0{noise_scale_w:.6f}\\0{max_chars}\\0{gap_ms}\\0{text}"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def synthesize_segmented(
        self,
        text: str,
        speaker: str,
        language: str,
        speed: float,
        noise_scale: float | None = None,
        noise_scale_w: float | None = None,
        max_chars: int = 140,
        gap_ms: int = 30,
    ) -> TTSResult:
        """Generate a full response using the same sentence-sized units as streaming.

        Generate Voice waits for all sentence-sized generations to finish, then
        returns one combined WAV. Because each unit goes through ``synthesize``
        with the same cache key as streaming, repeated Generate/Stream tests can
        reuse identical segment audio instead of producing unrelated long-form
        prosody.
        """
        if not self.is_loaded:
            raise RuntimeError("TTS model is not loaded")
        segments = split_sentences(text, max_chars=max_chars)
        if not segments:
            raise ValueError("Text is empty after TTS cleanup")
        noise_scale = self.noise_scale if noise_scale is None else self._validate_noise("noise_scale", noise_scale)
        noise_scale_w = self.noise_scale_w if noise_scale_w is None else self._validate_noise("noise_scale_w", noise_scale_w)
        gap_ms = max(0, int(gap_ms))

        if len(segments) == 1:
            result = self.synthesize(
                segments[0], speaker, language, speed, noise_scale, noise_scale_w
            )
            return TTSResult(
                audio=result.audio,
                sample_rate=result.sample_rate,
                generation_ms=result.generation_ms,
                audio_seconds=result.audio_seconds,
                rtf=result.rtf,
                device=result.device,
                dtype=result.dtype,
                gpu_memory_mb=result.gpu_memory_mb,
                cache_hit=result.cache_hit,
                preprocess_ms=result.preprocess_ms,
                inference_ms=result.inference_ms,
                text_cache_hit=result.text_cache_hit,
                segment_count=1,
                segmented=False,
            )

        full_key = self._segmented_cache_key(
            text, speaker, language, speed, noise_scale, noise_scale_w, max_chars, gap_ms
        )
        cached = self.cache.get(full_key)
        if cached is not None:
            return cached

        results: list[TTSResult] = []
        for segment in segments:
            results.append(
                self.synthesize(segment, speaker, language, speed, noise_scale, noise_scale_w)
            )

        sample_rate = results[0].sample_rate
        if any(result.sample_rate != sample_rate for result in results):
            raise RuntimeError("TTS segments returned different sample rates")
        audio = self.combine_audio_segments([result.audio for result in results], sample_rate, gap_ms)
        generation_ms = sum(result.generation_ms for result in results)
        preprocess_ms = sum(result.preprocess_ms for result in results)
        inference_ms = sum(result.inference_ms for result in results)
        audio_seconds = len(audio) / sample_rate
        cache_hit = all(result.cache_hit for result in results)
        text_cache_hit = all(result.text_cache_hit for result in results)
        gpu_mb = max(result.gpu_memory_mb for result in results)
        combined = TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            generation_ms=generation_ms,
            audio_seconds=audio_seconds,
            rtf=generation_ms / 1000.0 / audio_seconds if audio_seconds else 0.0,
            device=results[0].device,
            dtype=results[0].dtype,
            gpu_memory_mb=gpu_mb,
            cache_hit=cache_hit,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            text_cache_hit=text_cache_hit,
            segment_count=len(results),
            segmented=True,
        )
        self.cache.put(full_key, combined)
        return combined

    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str,
        speed: float,
        noise_scale: float | None = None,
        noise_scale_w: float | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("TTS model is not loaded")
        if speaker not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker}")
        if language not in LANGUAGE_MARKS:
            raise ValueError(f"Unsupported language: {language}")
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        noise_scale = self.noise_scale if noise_scale is None else self._validate_noise("noise_scale", noise_scale)
        noise_scale_w = self.noise_scale_w if noise_scale_w is None else self._validate_noise("noise_scale_w", noise_scale_w)

        assert self.hps is not None
        assert self.model is not None

        cache_key = self._cache_key(text, speaker, language, speed, noise_scale, noise_scale_w)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        with self._load_lock:
            started = time.perf_counter()
            text_started = time.perf_counter()
            stn, text_cache_hit = self._prepare_text_cached(text, language)
            preprocess_ms = (time.perf_counter() - text_started) * 1000.0

            # The tensors are tiny. Avoid pinning/async-transfer machinery here;
            # the CPU-side frontend is usually more expensive than this copy.
            x = stn.unsqueeze(0).to(self.device)
            x_lengths = torch.tensor([stn.numel()], dtype=torch.long, device=self.device)
            sid = torch.tensor([self.speakers[speaker]], dtype=torch.long, device=self.device)

            inference_started = time.perf_counter()
            with torch.inference_mode():
                compiled_infer = self._get_infer_callable()
                if compiled_infer is not None:
                    try:
                        if self.device.type == "cuda" and self.dtype == torch.float16:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                output = compiled_infer(
                                    x, x_lengths, sid, 1.0 / speed, noise_scale, noise_scale_w
                                )
                        else:
                            output = compiled_infer(
                                x, x_lengths, sid, 1.0 / speed, noise_scale, noise_scale_w
                            )
                        self.compile_active = True
                    except Exception as exc:
                        self.compile_error = f"compile execution failed: {type(exc).__name__}: {exc}"
                        self._compiled_infer = None
                        self.compile_active = False
                        output = self.model.infer(
                            x,
                            x_lengths,
                            sid=sid,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=1.0 / speed,
                        )[0][0, 0]
                elif self.device.type == "cuda" and self.dtype == torch.float16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.model.infer(
                            x,
                            x_lengths,
                            sid=sid,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=1.0 / speed,
                        )[0][0, 0]
                else:
                    output = self.model.infer(
                        x,
                        x_lengths,
                        sid=sid,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1.0 / speed,
                    )[0][0, 0]

            # .cpu() performs the required synchronization before the waveform
            # reaches NumPy, avoiding an extra cuda.synchronize() on every call.
            audio = output.detach().float().cpu().numpy()
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            elapsed = time.perf_counter() - started

        audio = self._normalize_audio(audio)
        sample_rate = int(self.hps.data.sampling_rate)
        audio_seconds = len(audio) / sample_rate
        gpu_mb = 0.0
        if self.device.type == "cuda":
            gpu_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        result = TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            generation_ms=elapsed * 1000.0,
            audio_seconds=audio_seconds,
            rtf=elapsed / audio_seconds if audio_seconds else 0.0,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            gpu_memory_mb=gpu_mb,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            cache_hit=False,
            text_cache_hit=text_cache_hit,
        )
        self.cache.put(cache_key, result)
        return result
