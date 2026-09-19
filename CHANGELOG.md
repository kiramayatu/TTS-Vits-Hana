# Changelog

## 1.0.1 — UI language + output loudness fix

- Fixed the duplicate Japanese language entries.
- `Japanese` now correctly wraps input with the `[JA]` marker required by the trained multilingual frontend.
- `日本語` remains accepted as a backward-compatible API alias.
- Removed `日本語` from the new UI option list so there is only one Japanese choice.
- Added conservative peak normalization (target -1 dBFS, max +24 dB gain) to address unusually quiet generated WAV output.
- Added regression tests for the language mapping and loudness normalization code.
