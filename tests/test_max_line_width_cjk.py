import unicodedata

import pytest

from whisper.utils import WriteSRT, WriteVTT


def display_width(text: str) -> int:
    """Columns a subtitle renderer gives *text* (CJK glyphs are two columns)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def chinese_result(text: str) -> dict:
    # split_tokens_on_unicode() emits one word per decoded code point for zh/ja/yue.
    words = [
        {"word": ch, "start": i * 0.3, "end": 0.3 + i * 0.3}
        for i, ch in enumerate(text)
    ]
    return {"segments": [{"start": 0.0, "end": 9.0, "text": text, "words": words}]}


def english_result(text: str) -> dict:
    words = [
        {"word": word + " ", "start": i * 0.5, "end": 0.5 + i * 0.5}
        for i, word in enumerate(text.split())
    ]
    return {"segments": [{"start": 0.0, "end": 9.0, "text": text, "words": words}]}


@pytest.mark.parametrize("writer", [WriteSRT, WriteVTT])
@pytest.mark.parametrize("max_line_width", [8, 10, 20])
def test_chinese_lines_respect_max_line_width(writer, max_line_width):
    result = chinese_result("今天天气不错，我们去公园散步吧。")

    subtitles = list(
        writer(False).iterate_result(
            result, {"max_line_width": max_line_width, "max_line_count": 2}
        )
    )

    lines = [line for _, _, block in subtitles for line in block.split("\n")]
    assert lines
    # len() counted code points, so a CJK line came out twice as wide as asked.
    for line in lines:
        assert display_width(line) <= max_line_width, (line, display_width(line))


@pytest.mark.parametrize("writer", [WriteSRT, WriteVTT])
def test_latin_lines_are_unchanged(writer):
    result = english_result("the quick brown fox jumps over the lazy dog")

    subtitles = list(
        writer(False).iterate_result(
            result, {"max_line_width": 20, "max_line_count": 2}
        )
    )

    lines = [line for _, _, block in subtitles for line in block.split("\n")]
    assert lines
    # ASCII glyphs are one column wide, so the limit keeps meaning code points.
    for line in lines:
        assert len(line) <= 20, (line, len(line))
