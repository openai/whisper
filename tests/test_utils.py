from whisper.utils import WriteSRT

WORDS = "The quick brown fox jumps over the lazy dog while the cat sleeps".split()


def make_result():
    words = [
        {"word": " " + word, "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": 0.9}
        for i, word in enumerate(WORDS)
    ]
    return {
        "segments": [
            {
                "id": 0,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": "".join(word["word"] for word in words),
                "words": words,
            }
        ]
    }


def test_max_words_per_line_has_no_effect_with_max_line_width():
    # the CLI documents --max_words_per_line as having "no effect with
    # --max_line_width" and warns accordingly, so the two combined must
    # produce the same subtitles as --max_line_width alone
    writer = WriteSRT(".")
    result = make_result()

    baseline = list(writer.iterate_result(result, max_line_width=42))
    combined = list(
        writer.iterate_result(result, max_line_width=42, max_words_per_line=4)
    )

    assert combined == baseline
    assert len(baseline) == 1


def test_max_words_per_line_without_max_line_width():
    writer = WriteSRT(".")

    subtitles = list(writer.iterate_result(make_result(), max_words_per_line=4))

    assert [text for _, _, text in subtitles] == [
        "The quick brown fox",
        "jumps over the lazy",
        "dog while the cat",
        "sleeps",
    ]
