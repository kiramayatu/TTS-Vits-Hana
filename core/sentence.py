"""Sentence cleanup and incremental segmentation for TTS/LLM streaming."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = set("。！？!?．")
_CLOSE_AFTER_END = set("」』】〕〉》）)]}") | {"'", '"', "’", "”", "»"}
_FALLBACK_BREAKS = set("、，, ")

_ROLE_PREFIX_RE = re.compile(r"^\s*(?:assistant|user|system|hana|ryza)\s*:\s*", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*|```")
_MARKDOWN_MARK_RE = re.compile(r"[*_~]{1,3}")
_BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?！？。])\1{3,}")


def clean_tts_text(text: str) -> str:
    """Remove common LLM-only formatting without changing normal wording."""
    text = text.replace("\u0000", " ")
    text = _CODE_FENCE_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _BULLET_RE.sub("", text)
    text = _ROLE_PREFIX_RE.sub("", text)
    text = _MARKDOWN_MARK_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _is_boundary(chars: list[str], index: int) -> bool:
    ch = chars[index]
    if ch in _SENTENCE_END:
        # Keep runs such as "！？" together as one sentence boundary.
        if index + 1 < len(chars) and chars[index + 1] in _SENTENCE_END:
            return False
        return True
    if ch == ".":
        prev_is_digit = index > 0 and chars[index - 1].isdigit()
        next_is_digit = index + 1 < len(chars) and chars[index + 1].isdigit()
        if prev_is_digit and next_is_digit:
            return False
        return index + 1 >= len(chars) or chars[index + 1].isspace()
    return False


def _split_long_chunk(chunk: str, max_chars: int) -> list[str]:
    if len(chunk) <= max_chars:
        return [chunk]

    pieces: list[str] = []
    remaining = chunk.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = -1
        for idx in range(len(window) - 1, max(15, len(window) // 2), -1):
            if window[idx] in _FALLBACK_BREAKS:
                cut = idx + 1
                break
        if cut <= 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_sentences(text: str, max_chars: int = 140) -> list[str]:
    """Split text into TTS-sized sentences while preserving punctuation."""
    cleaned = clean_tts_text(text)
    if not cleaned:
        return []
    if max_chars < 16:
        raise ValueError("max_chars must be at least 16")

    chars = list(cleaned)
    result: list[str] = []
    start = 0
    index = 0
    while index < len(chars):
        if _is_boundary(chars, index):
            end = index + 1
            while end < len(chars) and chars[end] in _CLOSE_AFTER_END:
                end += 1
            chunk = "".join(chars[start:end]).strip()
            if chunk:
                result.extend(_split_long_chunk(chunk, max_chars))
            start = end
            index = end
            continue
        index += 1

    if start < len(chars):
        chunk = "".join(chars[start:]).strip()
        if chunk:
            result.extend(_split_long_chunk(chunk, max_chars))
    return result


@dataclass
class SentenceSplitter:
    """Incremental splitter for partial LLM output chunks."""

    max_chars: int = 140

    def __post_init__(self) -> None:
        if self.max_chars < 16:
            raise ValueError("max_chars must be at least 16")
        self._buffer = ""

    @property
    def buffer(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> list[str]:
        if chunk:
            self._buffer += chunk
        if not self._buffer:
            return []

        emitted: list[str] = []
        chars = list(self._buffer)
        start = 0
        for index in range(len(chars)):
            if _is_boundary(chars, index):
                end = index + 1
                while end < len(chars) and chars[end] in _CLOSE_AFTER_END:
                    end += 1
                candidate = "".join(chars[start:end]).strip()
                if candidate:
                    emitted.extend(_split_long_chunk(clean_tts_text(candidate), self.max_chars))
                start = end
        self._buffer = "".join(chars[start:])
        return emitted

    def flush(self) -> list[str]:
        if not self._buffer.strip():
            self._buffer = ""
            return []
        result = _split_long_chunk(clean_tts_text(self._buffer), self.max_chars)
        self._buffer = ""
        return result

    def reset(self) -> None:
        self._buffer = ""
