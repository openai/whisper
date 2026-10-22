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
    "multilingual,tokens,text",
    [
        (True, [1041, 243, 230], "\ufffd안"),
        (False, [21253, 222, 222], "\ufffd退"),
    ],
)
def test_split_on_unicode_invalid_prefix(multilingual, tokens, text):
    tokenizer = get_tokenizer(multilingual=multilingual)

    assert tokenizer.decode_with_timestamps(tokens) == text
    assert tokenizer.split_tokens_on_unicode(tokens) == ([text], [tokens])
    assert tokenizer.split_to_word_tokens(tokens) == ([text], [tokens])


@pytest.mark.parametrize("multilingual", [True, False])
@pytest.mark.parametrize("text", ["\ufffd", "\ufffd안", "A\ufffdB", "安🙂"])
def test_split_on_unicode_byte_tokens(multilingual, text):
    tokenizer = get_tokenizer(multilingual=multilingual)
    expected_tokens = [
        [
            tokenizer.encoding.encode_single_token(bytes([byte]))
            for byte in char.encode()
        ]
        for char in text
    ]
    tokens = [token for group in expected_tokens for token in group]

    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert words == list(text)
    assert word_tokens == expected_tokens


@pytest.mark.parametrize("multilingual", [True, False])
@pytest.mark.parametrize(
    "data",
    [b"\xe9", b"\xe9\x80", b"\x80\x80", b"\xe9A", b"\xed\xa0\x80"],
)
def test_split_on_unicode_invalid_bytes(multilingual, data):
    tokenizer = get_tokenizer(multilingual=multilingual)
    tokens = [tokenizer.encoding.encode_single_token(bytes([byte])) for byte in data]

    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == data.decode("utf-8", errors="replace")
    assert [token for group in word_tokens for token in group] == tokens


@pytest.mark.parametrize("multilingual", [True, False])
def test_split_on_unicode_preserves_text_and_timestamps(multilingual):
    tokenizer = get_tokenizer(multilingual=multilingual)
    text = " hello 안녕하세요 世界 🙂 \ufffd"
    tokens = [tokenizer.timestamp_begin] + tokenizer.encode(text) + [tokenizer.eot]

    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert "".join(words) == tokenizer.decode_with_timestamps(tokens)
    assert [token for group in word_tokens for token in group] == tokens


@pytest.mark.parametrize("multilingual", [True, False])
@pytest.mark.parametrize("special", ["eot", "timestamp_begin", "sot"])
def test_split_on_unicode_incomplete_bytes_before_special(multilingual, special):
    tokenizer = get_tokenizer(multilingual=multilingual)
    incomplete = tokenizer.encoding.encode_single_token(b"\xe9")
    sentinel = getattr(tokenizer, special)
    suffix = tokenizer.encode(" hello")
    tokens = [incomplete, sentinel] + suffix

    words, word_tokens = tokenizer.split_to_word_tokens(tokens)

    assert words == ["\ufffd", tokenizer.decode_with_timestamps([sentinel]), " hello"]
    assert word_tokens == [[incomplete], [sentinel], suffix]
