## 2.3.1 – Unified Generate Voice playback
- Removed the separate Stream Sentences button from the web UI.
- Generate Voice now uses the shared sentence-sized generation pipeline and plays completed chunks progressively as one generation.
- The same generated chunks are combined into the final WAV for the audio player and Download WAV action.
- Stop now controls both in-progress generation and queued playback.

## 2.3.0 – Shared sentence-generation pipeline
- Generate Voice now splits text into the same sentence-sized chunks used by Stream Sentences, then combines the audio into one WAV.
- Shared segment audio/cache path keeps Generate and Stream behavior aligned.
- Added configurable inter-segment silence (`HANA_SEGMENT_GAP_MS`, default 30 ms).
- Streamed WAV download now preserves the same inter-segment gap.

## 1.2.3 — Stage 2.2 UI/profile + stream state hotfix

- Added browser-local voice profiles for Speed, Noise scale, Noise scale W, and Sentence max chars.
- Added Save, Load, Delete, and Reset to Defaults controls for voice profiles.
- Added descriptions for every voice/stream slider in the web UI.
- Fixed the Stream Sentences UI remaining disabled after the server reported `done`.
- The browser now closes the completed stream socket and restores Generate/Stream controls without requiring a page refresh.
- Stream playback source bookkeeping now removes completed audio sources.

## 1.2.1 — Stage 2.1 target-machine optimization

- Switched the automatic runtime dtype default to FP32 based on RTX 5060 Ti benchmark results; FP16 remains an explicit opt-in.
- Kept `torch.compile` opt-in and documented the measured regression on this legacy VITS workload.
- Added an LRU cache for CPU-side text/phoneme preprocessing to reduce repeated Japanese G2P work.
- Removed redundant CUDA synchronization around each synthesis request; the GPU-to-CPU waveform transfer already provides the required completion synchronization.
- Added preprocessing/inference timing metrics to the API/UI and benchmark output.
- Expanded the benchmark to use varied Japanese sentence lengths and report phase timings.
- Added regression tests for the FP32 default and text cache.


## 1.2.0 — Stage 2 runtime optimization

- Added optional CUDA FP16 autocast with safe FP32 model parameters.
- Removed generator weight normalization by default at startup for inference.
- Added a bounded single-worker FIFO TTS queue.
- Added byte-budgeted LRU audio caching.
- Added runtime optimization/queue/cache status reporting.
- Added `/cache/clear`.
- Added a benchmark script for comparing FP32, FP16, and optional `torch.compile` behavior on the target machine.
- Kept `torch.compile` disabled by default because dynamic text lengths may cause recompilation overhead.

# Changelog

## 1.0.1 — UI language + output loudness fix

- Fixed the duplicate Japanese language entries.
- `Japanese` now correctly wraps input with the `[JA]` marker required by the trained multilingual frontend.
- `日本語` remains accepted as a backward-compatible API alias.
- Removed `日本語` from the new UI option list so there is only one Japanese choice.
- Added conservative peak normalization (target -1 dBFS, max +24 dB gain) to address unusually quiet generated WAV output.
- Added regression tests for the language mapping and loudness normalization code.

## 1.2.2 — Stage 2.2 sentence streaming + voice controls

- Added Japanese-safe sentence splitting and incremental `SentenceSplitter` for partial LLM output.
- Added TTS text cleanup for common LLM formatting before sentence generation.
- Added `POST /text/split` for inspecting cleaned text and sentence boundaries.
- Added `WS /ws/tts` for incremental sentence-to-audio streaming.
- Streaming accepts repeated `append` chunks, emits completed sentences immediately, supports `flush`, and can cancel stale queued results.
- Added Web Audio scheduling in the lightweight UI so streamed sentence audio begins as soon as the first segment arrives and later segments are queued for playback.
- Added client-side combined WAV download for streamed segments without regenerating them.
- Added optional per-request `noise_scale` and `noise_scale_w` controls; cache keys include these inference parameters.
- Set the production server's default dtype explicitly to FP32 based on the RTX 5060 Ti benchmark.
- Kept `torch.compile` disabled by default.
- Added optional RMS loudness normalization (`HANA_LOUDNESS_MODE=rms`) while keeping peak normalization as the production default.
