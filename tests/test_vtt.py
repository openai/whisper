from copy import deepcopy
from io import StringIO

import pytest

from whisper.utils import WriteSRT, WriteVTT


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<name>", "&lt;name&gt;"),
        ("&lt; &amp; &", "&amp;lt; &amp;amp; &amp;"),
        ("<u>name</u>", "&lt;u&gt;name&lt;/u&gt;"),
        ("a > b", "a &gt; b"),
        ('"quoted" text', '"quoted" text'),
        ("hello 世界", "hello 世界"),
    ],
)
def test_vtt_escapes_segment_text(text, expected):
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": text}]}
    original = deepcopy(result)
    output = StringIO()
    WriteVTT(".").write_result(result, output)
    assert output.getvalue() == f"WEBVTT\n\n00:00.000 --> 00:01.000\n{expected}\n\n"
    assert result == original


@pytest.mark.parametrize("highlight", [False, True])
def test_vtt_escapes_words_before_highlighting(highlight):
    words = [
        {"word": " <u>name</u>", "start": 0.0, "end": 0.5},
        {"word": " &lt;", "start": 1.0, "end": 1.5},
    ]
    result = {"segments": [{"start": 0.0, "end": 1.5, "words": words}]}
    original = deepcopy(result)
    cues = list(WriteVTT(".").iterate_result(result, highlight_words=highlight))
    text = "&lt;u&gt;name&lt;/u&gt; &amp;lt;"
    expected = (
        [
            ("00:00.000", "00:00.500", "<u>&lt;u&gt;name&lt;/u&gt;</u> &amp;lt;"),
            ("00:00.500", "00:01.000", text),
            ("00:01.000", "00:01.500", "&lt;u&gt;name&lt;/u&gt; <u>&amp;lt;</u>"),
        ]
        if highlight
        else [("00:00.000", "00:01.500", text)]
    )
    assert cues == expected
    assert result == original


@pytest.mark.parametrize(
    "writer_class, expected",
    [(WriteVTT, "&lt;x&gt; &amp;\ny"), (WriteSRT, "<x> &\ny")],
)
def test_subtitle_wrapping_uses_unescaped_text(writer_class, expected):
    words = [
        {"word": " <x>", "start": 0.0, "end": 1.0},
        {"word": " &", "start": 1.0, "end": 2.0},
        {"word": " y", "start": 2.0, "end": 3.0},
    ]
    result = {"segments": [{"start": 0.0, "end": 3.0, "words": words}]}
    original = deepcopy(result)
    cues = list(writer_class(".").iterate_result(result, max_line_width=5))
    assert len(cues) == 1
    assert cues[0][2] == expected
    assert result == original


@pytest.mark.parametrize("with_words", [False, True])
def test_srt_keeps_existing_text_format(with_words):
    text = "<u>name</u> &lt;"
    segment = {"start": 0.0, "end": 1.0, "text": text}
    if with_words:
        segment["words"] = [{"word": text, "start": 0.0, "end": 1.0}]
    result = {"segments": [segment]}
    assert list(WriteSRT(".").iterate_result(result)) == [
        ("00:00:00,000", "00:00:01,000", text)
    ]
