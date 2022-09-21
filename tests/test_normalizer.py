import pytest

from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers.english import EnglishNumberNormalizer, EnglishSpellingNormalizer


@pytest.mark.parametrize("std", [EnglishNumberNormalizer(), EnglishTextNormalizer()])
def test_number_normalizer(std):
    assert std("two") == "2"
    assert std("thirty one") == "31"
    assert std("five twenty four") == "524"
    assert std("nineteen ninety nine") == "1999"
    assert std("twenty nineteen") == "2019"

    assert std("two point five million") == "2500000"
    assert std("four point two billions") == "4200000000s"
    assert std("200 thousand") == "200000"
    assert std("200 thousand dollars") == "$200000"
    assert std("$20 million") == "$20000000"
    assert std("€52.4 million") == "€52400000"
    assert std("£77 thousands") == "£77000s"

    assert std("two double o eight") == "2008"

    assert std("three thousand twenty nine") == "3029"
    assert std("forty three thousand two hundred sixty") == "43260"
    assert std("forty three thousand two hundred and sixty") == "43260"

    assert std("nineteen fifties") == "1950s"
    assert std("thirty first") == "31st"
    assert std("thirty three thousand and three hundred and thirty third") == "33333rd"

    assert std("three billion") == "3000000000"
    assert std("millions") == "1000000s"

    assert std("july third twenty twenty") == "july 3rd 2020"
    assert std("august twenty sixth twenty twenty one") == "august 26th 2021"
    assert std("3 14") == "3 14"
    assert std("3.14") == "3.14"
    assert std("3 point 2") == "3.2"
    assert std("3 point 14") == "3.14"
    assert std("fourteen point 4") == "14.4"
    assert std("two point two five dollars") == "$2.25"
    assert std("two hundred million dollars") == "$200000000"
    assert std("$20.1 million") == "$20100000"

    assert std("ninety percent") == "90%"
    assert std("seventy six per cent") == "76%"

    assert std("double oh seven") == "007"
    assert std("double zero seven") == "007"
    assert std("nine one one") == "911"
    assert std("nine double one") == "911"
    assert std("one triple oh one") == "10001"

    assert std("two thousandth") == "2000th"
    assert std("thirty two thousandth") == "32000th"

    assert std("minus 500") == "-500"
    assert std("positive twenty thousand") == "+20000"

    assert std("two dollars and seventy cents") == "$2.70"
    assert std("3 cents") == "¢3"
    assert std("$0.36") == "¢36"
    assert std("three euros and sixty five cents") == "€3.65"

    assert std("three and a half million") == "3500000"
    assert std("forty eight and a half dollars") == "$48.5"
    assert std("b747") == "b 747"
    assert std("10 th") == "10th"
    assert std("10th") == "10th"


def test_spelling_normalizer():
    std = EnglishSpellingNormalizer()

    assert std("mobilisation") == "mobilization"
    assert std("cancelation") == "cancellation"


def test_text_normalizer():
    std = EnglishTextNormalizer()
    assert std("Let's") == "let us"
    assert std("he's like") == "he is like"
    assert std("she's been like") == "she has been like"
    assert std("10km") == "10 km"
    assert std("RC232") == "rc 232"

    assert (
        std("Mr. Park visited Assoc. Prof. Kim Jr.")
        == "mister park visited associate professor kim junior"
    )
