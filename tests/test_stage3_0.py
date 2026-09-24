from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vram_status_is_visible_in_ui():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="gpuInfo"' in html
    assert "memory_allocated_mb" in js
    assert "memory_reserved_mb" in js
    assert "memory_total_mb" in js


def test_stop_handles_connecting_socket_race():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "if (runId !== streamGeneration || settled)" in js
    assert "socket.close(1000, 'user stop')" in js
    assert "activeGeneration" in js
    assert "readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING" in js


def test_stale_decoded_audio_is_not_scheduled_after_stop():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "scheduleStreamBlob(event.data, runId)" in js
    assert "if (runId !== streamGeneration || settled) return" in js
    assert "if (!activeGeneration || activeGeneration.runId !== runId) return;" in js


def test_runtime_status_polls_without_reloading_voice_options():
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "setInterval(loadStatus, 2000)" in js
    assert "loadVoiceOptions();" in js


def test_stage_3_0_does_not_change_fp32_default():
    server = (ROOT / "api" / "server.py").read_text(encoding="utf-8")
    assert 'HANA_DTYPE", "fp32"' in server
