import re
import unicodedata
from functools import partial

import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str, preserve_marks: bool = False):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics

    If `preserve_marks` is True, characters in the Unicode "Mark" categories
    (Mn, Mc, Me) are kept instead of being replaced with a space. This matters for
    scripts in which marks are part of the spelling of a word, such as the Brahmic
    scripts (Devanagari, Bengali, Tamil, Malayalam, ...), Thai, Thaana, or Arabic
    and Hebrew text with vowel points: replacing those marks with spaces splits
    every word into fragments.
    """
    categories = "SP" if preserve_marks else "MSP"
    return "".join(
        " " if unicodedata.category(c)[0] in categories else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(
        self,
        remove_diacritics: bool = False,
        split_letters: bool = False,
        preserve_marks: bool = False,
    ):
        if remove_diacritics and preserve_marks:
            raise ValueError(
                "`preserve_marks` cannot be combined with `remove_diacritics`"
            )

        if remove_diacritics:
            self.clean = remove_symbols_and_diacritics
        elif preserve_marks:
            self.clean = partial(remove_symbols, preserve_marks=True)
        else:
            self.clean = remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s
