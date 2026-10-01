import pytest

from whisper.utils import WriteSRT, WriteVTT


@pytest.mark.parametrize("writer_class", [WriteSRT, WriteVTT])
@pytest.mark.parametrize("use_options", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_negative_word_limit(writer_class, use_options, empty):
    writer = writer_class(".")
    limits = {"max_words_per_line": -1}
    kwargs = {"options": limits} if use_options else limits
    segments = (
        []
        if empty
        else [{"start": 0.0, "words": [{"start": 0.0, "end": 1.0, "word": "one"}]}]
    )
    with pytest.raises(ValueError, match="max_words_per_line"):
        list(writer.iterate_result({"segments": segments}, **kwargs))


@pytest.mark.parametrize("writer_class", [WriteSRT, WriteVTT])
@pytest.mark.parametrize(
    "limit,expected",
    [(None, ["one two"]), (0, ["one two"]), (1, ["one", "two"]), (2, ["one two"])],
)
def test_word_limit(writer_class, limit, expected):
    result = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "one two",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "one"},
                    {"start": 1.0, "end": 2.0, "word": " two"},
                ],
            }
        ]
    }
    subtitles = writer_class(".").iterate_result(result, max_words_per_line=limit)
    assert [text for _, _, text in subtitles] == expected
