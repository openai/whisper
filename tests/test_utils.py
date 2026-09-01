"""Tests for pure-Python helpers in whisper/utils.py."""
import importlib.util

import pytest


def _load_utils():
    spec = importlib.util.spec_from_file_location(
        "whisper_utils",
        __file__.replace("tests/test_utils.py", "whisper/utils.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


utils = _load_utils()


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


def test_format_timestamp_seconds_only():
    assert utils.format_timestamp(90.5) == "01:30.500"


def test_format_timestamp_always_include_hours():
    assert utils.format_timestamp(90.5, always_include_hours=True) == "00:01:30.500"


def test_format_timestamp_comma_marker():
    assert utils.format_timestamp(3661.001, decimal_marker=",") == "01:01:01,001"


def test_format_timestamp_zero():
    assert utils.format_timestamp(0.0) == "00:00.000"


# ---------------------------------------------------------------------------
# get_start / get_end  — without 'words' key (regression: KeyError)
# ---------------------------------------------------------------------------


def test_get_start_returns_segment_start_when_words_absent():
    """Segments without a 'words' key must not raise KeyError."""
    segments = [{"start": 1.5, "end": 4.0, "text": "hello"}]
    assert utils.get_start(segments) == 1.5


def test_get_end_returns_segment_end_when_words_absent():
    """Segments without a 'words' key must not raise KeyError."""
    segments = [{"start": 1.5, "end": 4.0, "text": "hello"}]
    assert utils.get_end(segments) == 4.0


def test_get_start_returns_none_for_empty_segments():
    assert utils.get_start([]) is None


def test_get_end_returns_none_for_empty_segments():
    assert utils.get_end([]) is None


def test_get_start_prefers_first_word_timestamp():
    segments = [
        {
            "start": 1.0,
            "end": 5.0,
            "text": "hi",
            "words": [
                {"word": "hi", "start": 1.2, "end": 1.8},
            ],
        }
    ]
    assert utils.get_start(segments) == 1.2


def test_get_end_prefers_last_word_timestamp():
    segments = [
        {
            "start": 1.0,
            "end": 5.0,
            "text": "hi there",
            "words": [
                {"word": "hi", "start": 1.2, "end": 1.8},
                {"word": " there", "start": 2.0, "end": 2.9},
            ],
        }
    ]
    assert utils.get_end(segments) == 2.9


def test_get_start_with_empty_words_list():
    """Empty 'words' list falls back to segment start."""
    segments = [{"start": 0.5, "end": 2.0, "text": "x", "words": []}]
    assert utils.get_start(segments) == 0.5


def test_get_end_with_empty_words_list():
    """Empty 'words' list falls back to segment end."""
    segments = [{"start": 0.5, "end": 2.0, "text": "x", "words": []}]
    assert utils.get_end(segments) == 2.0
