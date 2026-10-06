"""Regression checks for decimal currency amounts below one unit."""

from decimal import Decimal

import pytest

from whisper.normalizers.english import EnglishNumberNormalizer, EnglishTextNormalizer


@pytest.mark.parametrize("currency", ["$", "£", "€"])
@pytest.mark.parametrize("fraction", ["1", "2", "5", "9"])
def test_single_digit_fraction_preserves_its_numeric_value(currency, fraction):
    normalize = EnglishNumberNormalizer()
    expected = f"¢{int(Decimal('0.' + fraction) * 100)}"
    assert normalize(f"{currency}0.{fraction}") == expected
    assert normalize(f"{currency}0.{fraction}0") == expected


@pytest.mark.parametrize("normalizer", [EnglishNumberNormalizer, EnglishTextNormalizer])
@pytest.mark.parametrize(
    "amount,cents", [("one", "10"), ("two", "20"), ("five", "50"), ("nine", "90")]
)
def test_spoken_decimal_currency_matches_spoken_cents(normalizer, amount, cents):
    normalize = normalizer()
    assert normalize(f"zero point {amount} dollars") == f"¢{cents}"
    assert normalize(f"zero point {amount} zero dollars") == f"¢{cents}"


@pytest.mark.parametrize("currency", ["$", "£", "€"])
def test_all_two_digit_fractions_keep_existing_values(currency):
    normalize = EnglishNumberNormalizer()
    for cents in range(1, 100):
        assert normalize(f"{currency}0.{cents:02d}") == f"¢{cents}"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("$0", "$0"),
        ("$0.00", "$0"),
        ("$1.5", "$1.5"),
        ("$0.005", "$0.005"),
        ("£12.50", "£12.50"),
        ("five cents", "¢5"),
        ("fifty cents", "¢50"),
    ],
)
def test_other_currency_forms_keep_their_existing_behavior(value, expected):
    assert EnglishNumberNormalizer()(value) == expected


def test_cent_extraction_requires_a_literal_decimal_separator():
    normalize = EnglishNumberNormalizer()
    for value in ("$0x5", "£0 5", "€025"):
        assert normalize.postprocess(value) == value
