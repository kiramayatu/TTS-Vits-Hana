"""Text frontend for the existing Hana VITS multilingual symbol set."""
from __future__ import annotations

from torch import LongTensor

import commons
from text import text_to_sequence

# Canonical UI/API language options. The old project accidentally exposed both
# "Japanese" and "日本語" even though only the latter carried the required
# [JA] marker. Keep the Japanese alias for API backward compatibility, but
# expose only one Japanese option in the new UI.
LANGUAGE_MARKS = {
    "Japanese": "[JA]",
    "日本語": "[JA]",  # legacy alias
    "Chinese": "[ZH]",
    "简体中文": "[ZH]",  # legacy alias
    "English": "[EN]",
    "Mix": "",
}

# What the browser UI presents. These labels correspond to actual input
# languages, not alternate display spellings of the same language.
LANGUAGES = ["Japanese", "Chinese", "English", "Mix"]


def prepare_text(text: str, language: str, hps) -> LongTensor:
    prefix = LANGUAGE_MARKS.get(language)
    if prefix is None:
        raise ValueError(f"Unsupported language: {language}")

    prepared = text if not prefix else f"{prefix}{text}{prefix}"
    sequence = text_to_sequence(prepared, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        sequence = commons.intersperse(sequence, 0)
    if not sequence:
        raise ValueError(
            "Text produced an empty phoneme sequence. "
            "For Japanese, select 'Japanese' and enter Japanese text (kana/kanji)."
        )
    return LongTensor(sequence)
