import pytest

from whisper.utils import WriteVTT


@pytest.mark.parametrize(
    "next_start, expected_text",
    [
        (3.5, ["hello again"]),  # 2.5 seconds of silence
        (4.0, ["hello again"]),  # exactly the three-second threshold
        (4.1, ["hello", "again"]),  # more than three seconds of silence
    ],
)
def test_subtitle_break_uses_gap_between_words(next_start, expected_text):
    words = [
        {"start": 0.0, "end": 1.0, "word": " hello"},
        {"start": next_start, "end": next_start + 0.5, "word": " again"},
    ]
    result = {"segments": [{"start": 0.0, "end": next_start + 0.5, "words": words}]}

    cues = WriteVTT(".").iterate_result(result, max_line_width=40, max_line_count=2)

    assert [text for _, _, text in cues] == expected_text
