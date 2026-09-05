import pytest

from whisper.utils import WriteSRT, WriteVTT


@pytest.mark.parametrize("writer_class", [WriteVTT, WriteSRT])
@pytest.mark.parametrize("highlight", [False, True])
def test_word_subtitles_escape_timestamp_arrows(writer_class, highlight):
    word = {"word": " left --> right", "start": 0.0, "end": 1.0}
    result = {
        "segments": [{"start": 0.0, "end": 1.0, "text": word["word"], "words": [word]}]
    }
    cues = list(writer_class(".").iterate_result(result, highlight_words=highlight))
    assert cues
    assert all("-->" not in text for _, _, text in cues)
    assert "left -> right" in cues[0][2]
    assert word["word"] == " left --> right"
