from copy import deepcopy

import pytest

from whisper.utils import WriteSRT, WriteVTT


@pytest.mark.parametrize("writer_class", [WriteVTT, WriteSRT])
@pytest.mark.parametrize("highlight", [False, True])
def test_word_subtitles_escape_timestamp_arrows(writer_class, highlight):
    word = {"word": " left --> right", "start": 0.0, "end": 1.0}
    result = {
        "segments": [{"start": 0.0, "end": 1.0, "text": word["word"], "words": [word]}]
    }
    original = deepcopy(result)
    cues = list(writer_class(".").iterate_result(result, highlight_words=highlight))
    assert cues
    assert all("-->" not in text for _, _, text in cues)
    assert "left -> right" in cues[0][2]
    assert result == original
    expected_times = (
        ("00:00.000", "00:01.000")
        if writer_class is WriteVTT
        else ("00:00:00,000", "00:00:01,000")
    )
    assert cues[0][:2] == expected_times
