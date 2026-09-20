from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_python_files_parse():
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_required_files_exist():
    for rel in [
        "models/G_latest.pth",
        "config/fine.json",
        "api/server.py",
        "core/engine.py",
        "web/index.html",
        "web/app.js",
        "scripts/benchmark.py",
    ]:
        assert (ROOT / rel).exists(), rel


def test_production_runtime_does_not_import_gradio():
    for rel in ["api/server.py", "core/engine.py", "core/text.py"]:
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "import gradio" not in source
        assert "from gradio" not in source


def test_japanese_language_aliases_map_to_ja():
    from core.text import LANGUAGE_MARKS, LANGUAGES

    assert LANGUAGE_MARKS["Japanese"] == "[JA]"
    assert LANGUAGE_MARKS["日本語"] == "[JA]"
    assert "Japanese" in LANGUAGES
    assert "日本語" not in LANGUAGES


def test_audio_normalization_code_present():
    source = (ROOT / "core/engine.py").read_text(encoding="utf-8")
    assert "target_peak = 10.0 ** (-1.0 / 20.0)" in source
    assert "max_gain = 10.0 ** (24.0 / 20.0)" in source


def test_stage2_controls_present():
    source = (ROOT / "core/engine.py").read_text(encoding="utf-8")
    assert "torch.autocast" in source
    assert "remove_weight_norm" in source
    assert "class AudioCache" in source
    assert "HANA_TORCH_COMPILE" in (ROOT / "api/server.py").read_text(encoding="utf-8")


def test_lru_cache_eviction():
    import numpy as np
    from core.engine import AudioCache, TTSResult

    cache = AudioCache(max_items=1, max_bytes=1024)
    result = TTSResult(np.zeros(20, dtype=np.float32), 22050, 1.0, 0.01, 0.1, "cpu", "float32", 0.0)
    cache.put("a", result)
    assert cache.get("a") is not None
    cache.put("b", result)
    assert cache.get("a") is None
    assert cache.get("b") is not None


def test_cache_hit_is_marked():
    import numpy as np
    from core.engine import AudioCache, TTSResult

    cache = AudioCache(max_items=2, max_bytes=1024)
    result = TTSResult(np.ones(10, dtype=np.float32), 22050, 10.0, 0.01, 1.0, "cpu", "float32", 0.0)
    cache.put("a", result)
    hit = cache.get("a")
    assert hit is not None
    assert hit.cache_hit is True
    assert hit.generation_ms == 0.0


def test_auto_dtype_prefers_fp32_for_legacy_vits():
    from core.engine import HanaVITSEngine
    engine = HanaVITSEngine(ROOT / "models/G_latest.pth", ROOT / "config/fine.json", "cpu", "auto",
                            cache_items=0, cache_max_mb=0)
    assert engine.dtype.__str__() == "torch.float32"


def test_text_sequence_cache_lru():
    import torch
    from core.engine import TextSequenceCache
    cache = TextSequenceCache(max_items=1)
    value = torch.tensor([1, 2, 3], dtype=torch.long)
    cache.put("a", value)
    assert cache.get("a") is value
    cache.put("b", value)
    assert cache.get("a") is None
    assert cache.get("b") is value
