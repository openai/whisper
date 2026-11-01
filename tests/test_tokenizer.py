import pytest

from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize("multilingual", [True, False])
def test_tokenizer(multilingual):
    tokenizer = get_tokenizer(multilingual=False)
    assert tokenizer.sot in tokenizer.sot_sequence
    assert len(tokenizer.all_language_codes) == len(tokenizer.all_language_tokens)
    assert all(c < tokenizer.timestamp_begin for c in tokenizer.all_language_tokens)


def test_multilingual_tokenizer():
    gpt2_tokenizer = get_tokenizer(multilingual=False)
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    text = "다람쥐 헌 쳇바퀴에 타고파"
    gpt2_tokens = gpt2_tokenizer.encode(text)
    multilingual_tokens = multilingual_tokenizer.encode(text)

    assert gpt2_tokenizer.decode(gpt2_tokens) == text
    assert multilingual_tokenizer.decode(multilingual_tokens) == text
    assert len(gpt2_tokens) > len(multilingual_tokens)


def test_split_on_unicode():
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = multilingual_tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]


@pytest.mark.parametrize(
    "language, text, expected",
    [
        (
            "en",
            " It rose 3.5 to $1,234.50",
            [" It", " rose", " 3.5", " to", " $1,234.50"],
        ),
        (
            "en",
            " At 10:30 on example.com.",
            [" At", " 10:30", " on", " example.com", "."],
        ),
        ("en", " Hello, world!", [" Hello", ",", " world", "!"]),
        ("en", " It rose 4% today.", [" It", " rose", " 4", "%", " today", "."]),
        (
            "ko",
            ' 그는 "안녕"이라고 말했다.',
            [" 그는", ' "안녕"이라고', " 말했다", "."],
        ),
    ],
)
def test_split_on_spaces_only_separates_trailing_punctuation(language, text, expected):
    tokenizer = get_tokenizer(multilingual=True, language=language, task="transcribe")

    tokens = tokenizer.encode(text) + [tokenizer.eot]
    words, word_tokens = tokenizer.split_tokens_on_spaces(tokens)

    assert words == expected + ["<|endoftext|>"]
    assert [token for group in word_tokens for token in group] == tokens
