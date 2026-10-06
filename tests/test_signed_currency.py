"""Currency names must not erase an already recognized numeric sign."""

import pytest

from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers.english import EnglishNumberNormalizer


@pytest.mark.parametrize("normalizer", [EnglishNumberNormalizer, EnglishTextNormalizer])
@pytest.mark.parametrize(
    "word,sign", [("minus", "-"), ("negative", "-"), ("plus", "+"), ("positive", "+")]
)
@pytest.mark.parametrize(
    "currency,symbol",
    [("dollars", "$"), ("pounds", "£"), ("euros", "€"), ("cents", "¢")],
)
def test_currency_names_preserve_numeric_signs(
    normalizer, word, sign, currency, symbol
):
    normalize = normalizer()
    assert normalize(f"{word} five {currency}") == f"{sign}{symbol}5"
    assert normalize(f"{word} five {currency}") != normalize(f"five {currency}")


@pytest.mark.parametrize("normalizer", [EnglishNumberNormalizer, EnglishTextNormalizer])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("minus one dollar", "-$1"),
        ("negative two point two five euros", "-€2.25"),
        ("plus five pounds and twenty cents", "+£5.20"),
        ("minus five dollars and twenty cents", "-$5.20"),
        ("negative two million dollars", "-$2000000"),
        ("minus five dollars then seven euros", "-$5 then €7"),
        ("five dollars", "$5"),
        ("$5 dollars", "$5"),
        ("minus five", "-5"),
        ("positive one percent", "+1%"),
    ],
)
def test_amount_composition_and_existing_behavior(normalizer, text, expected):
    assert normalizer()(text) == expected
