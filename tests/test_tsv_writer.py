import io

import pytest

from whisper.utils import WriteTSV


@pytest.mark.parametrize(
    "text", ["first\nsecond", "first\rsecond", "first\r\nsecond", "first\tsecond"]
)
def test_tsv_keeps_segment_text_in_one_row(text):
    output = io.StringIO()
    WriteTSV(".").write_result(
        {"segments": [{"start": 0.0, "end": 1.25, "text": text}]}, output
    )
    rows = output.getvalue().splitlines()
    assert len(rows) == 2
    assert rows[0] == "start\tend\ttext"
    assert rows[1].split("\t")[:2] == ["0", "1250"]
    assert rows[1].split("\t")[2].split() == ["first", "second"]
