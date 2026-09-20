from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_ui_exists():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    for element_id in ("profileName", "saveProfile", "resetVoice", "profileSelect", "loadProfile", "deleteProfile"):
        assert f'id="{element_id}"' in html


def test_slider_descriptions_exist():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "Controls speech variation" in html
    assert "Controls stochastic timing/prosody variation" in html
    assert "Maximum characters per sentence generation chunk" in html
    assert "Speech speed" in html


def test_profile_storage_and_defaults_exist():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "hanaVits.voiceProfiles.v1" in js
    assert "DEFAULT_VOICE_PROFILE" in js
    assert "applyVoiceProfile(DEFAULT_VOICE_PROFILE)" in js


def test_generate_uses_shared_segment_engine():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    server = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
    engine = (ROOT / "core" / "engine.py").read_text(encoding="utf-8")
    assert "sentence_max" in js
    assert "X-TTS-Segments" in js
    assert "Maximum characters per sentence generation chunk" in html
    assert "submit_grouped" in server
    assert "synthesize_segmented" in engine



def test_generate_endpoint_uses_grouped_queue(monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient
    import api.server as server
    from core.engine import TTSResult

    class FakeQueue:
        def start(self):
            pass

        def submit_grouped(self, *args):
            assert args[0] == "一つ目。二つ目。"
            assert args[6] == 140
            assert args[7] == server.SEGMENT_GAP_MS
            return TTSResult(
                audio=np.zeros(220, dtype=np.float32),
                sample_rate=22000,
                generation_ms=2.0,
                audio_seconds=0.01,
                rtf=0.2,
                device="cpu",
                dtype="float32",
                gpu_memory_mb=0.0,
                segment_count=2,
                segmented=True,
            )

    monkeypatch.setattr(server.engine, "load", lambda: None)
    monkeypatch.setattr(server, "inference_queue", FakeQueue())
    with TestClient(server.app) as client:
        response = client.post("/tts", json={"text": "一つ目。二つ目。"})
    assert response.status_code == 200
    assert response.headers["X-TTS-Segments"] == "2"
    assert response.headers["X-TTS-Segmented"] == "1"


def test_generate_uses_internal_progressive_playback():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "generateWithProgressivePlayback" in js
    assert "new WebSocket" in js
    assert "await scheduleStreamBlob" in js
    assert "combineWavs(streamBlobs)" in js
    assert 'id="stream"' not in html
    assert 'id="downloadStream"' not in html
    assert 'Stream Sentences' not in html
    assert 'id="generate"' in html


def test_shared_controls_have_single_generate_path():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "noise_scale: Number($('noiseScale').value)" in js
    assert "noise_scale_w: Number($('noiseScaleW').value)" in js
    assert "max_chars: Number($('sentenceMax').value)" in js
    assert "action: 'append'" in js
    assert "action: 'flush'" in js


def test_engine_combines_audio_segments():
    import numpy as np
    from core.engine import HanaVITSEngine

    a = np.ones(4, dtype=np.float32)
    b = np.ones(2, dtype=np.float32) * 2
    out = HanaVITSEngine.combine_audio_segments([a, b], sample_rate=100, gap_ms=20)
    assert out.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0]


def test_generate_restores_controls_after_completion_or_stop():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "button.disabled = false;" in js
    assert "$('stop').disabled = true;" in js
    assert "Generation/playback stopped." in js
    assert "generation complete" in js
