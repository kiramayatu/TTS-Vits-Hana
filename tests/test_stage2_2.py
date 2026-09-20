from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_japanese_sentence_split():
    from core.sentence import split_sentences

    text = "こんにちは、マスター。今日はどうしますか！？えへへ……楽しみですね。"
    assert split_sentences(text) == [
        "こんにちは、マスター。",
        "今日はどうしますか！？",
        "えへへ……楽しみですね。",
    ]


def test_sentence_split_preserves_closing_quote():
    from core.sentence import split_sentences

    assert split_sentences('「本当ですか？」それなら、うれしいです。') == [
        '「本当ですか？」',
        'それなら、うれしいです。',
    ]


def test_sentence_split_avoids_decimal_break():
    from core.sentence import split_sentences

    assert split_sentences("バージョン2.14.0を使っています。次に進みます。") == [
        "バージョン2.14.0を使っています。",
        "次に進みます。",
    ]


def test_incremental_splitter():
    from core.sentence import SentenceSplitter

    splitter = SentenceSplitter(max_chars=140)
    assert splitter.feed("こんにちは、マスター。") == ["こんにちは、マスター。"]
    assert splitter.feed("今日はどうします") == []
    assert splitter.feed("か？") == ["今日はどうしますか？"]
    assert splitter.flush() == []


def test_long_fragment_splits_at_comma():
    from core.sentence import split_sentences

    text = "これはとても長い文章なので、読み上げるときには一度に全部処理せず、自然なところで区切って、次の文章へ進めるようにしたいです。"
    parts = split_sentences(text, max_chars=50)
    assert len(parts) >= 2
    assert all(len(part) <= 50 for part in parts)


def test_tts_cleanup_removes_role_and_markdown():
    from core.sentence import clean_tts_text

    cleaned = clean_tts_text("Hana: **こんにちは** [OpenAI](https://openai.com)")
    assert cleaned == "こんにちは OpenAI"


def test_engine_cache_key_contains_noise_parameters():
    from core.engine import HanaVITSEngine

    a = HanaVITSEngine._cache_key("こんにちは。", "Hana", "Japanese", 1.0, 0.667, 0.8)
    b = HanaVITSEngine._cache_key("こんにちは。", "Hana", "Japanese", 1.0, 0.3, 0.5)
    assert a != b


def test_server_exposes_stage22_routes():
    from api.server import app

    paths = {route.path for route in app.routes}
    assert "/text/split" in paths
    assert "/ws/tts" in paths


def test_websocket_stream_emits_segments_in_order(monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient
    import api.server as server
    from core.engine import TTSResult

    def fake_submit(request):
        text = request.text
        samples = np.zeros(220, dtype=np.float32)
        return TTSResult(
            audio=samples,
            sample_rate=22000,
            generation_ms=1.0,
            audio_seconds=0.01,
            rtf=0.1,
            device="cpu",
            dtype="float32",
            gpu_memory_mb=0.0,
            cache_hit=False,
            preprocess_ms=0.1,
            inference_ms=0.9,
            text_cache_hit=False,
        )

    monkeypatch.setattr(server, "_submit_request", fake_submit)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_json({"action": "start", "speaker": "Hana", "language": "Japanese", "speed": 1.0})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"action": "append", "text": "一つ目。二つ目。"})
            ws.send_json({"action": "flush"})

            seen = []
            binary = 0
            while True:
                message = ws.receive()
                if message.get("text") is not None:
                    payload = __import__("json").loads(message["text"])
                    if payload["type"] == "segment":
                        seen.append(payload["text"])
                    elif payload["type"] == "done":
                        break
                elif message.get("bytes") is not None:
                    binary += 1

            assert seen == ["一つ目。", "二つ目。"]
            assert binary == 2


def test_loudness_modes_are_valid():
    from core.engine import HanaVITSEngine

    base = dict(model_path=ROOT / "models/G_latest.pth", config_path=ROOT / "config/fine.json", device="cpu", dtype="fp32", cache_items=0, cache_max_mb=0)
    for mode in ("peak", "rms", "off"):
        engine = HanaVITSEngine(**base, loudness_mode=mode)
        assert engine.loudness_mode == mode
