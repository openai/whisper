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
    "language, expected",
    [
        ("en", "en"),
        ("EN", "en"),
        ("english", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("zh-CN", "zh"),
        ("zh_Hans", "zh"),
        ("pt-BR", "pt"),
        ("ko-KR", "ko"),
        ("mandarin", "zh"),
    ],
)
def test_language_locale_tags(language, expected):
    tokenizer = get_tokenizer(multilingual=True, language=language)
    assert tokenizer.language == expected


def test_language_locale_tags_rejected_when_unknown():
    with pytest.raises(ValueError, match="Unsupported language: xx-us"):
        get_tokenizer(multilingual=True, language="xx-US")


def test_english_only_tokenizer_accepts_locale_tags():
    # English-only models ignore the language, but a locale tag should not raise.
    tokenizer = get_tokenizer(multilingual=False, language="en-US")
    assert tokenizer.language is None
