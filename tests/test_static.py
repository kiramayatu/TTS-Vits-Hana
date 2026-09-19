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
