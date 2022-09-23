import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import GPT2TokenizerFast

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


@dataclass(frozen=True)
class Tokenizer:
    """A thin wrapper around `GPT2TokenizerFast` providing quick access to special tokens"""

    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: Union[int, List[int], np.ndarray, torch.Tensor], **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, tokens) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    @property
    @lru_cache()
    def eot(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    @lru_cache()
    def sot(self) -> int:
        return self._get_single_token_id("<|startoftranscript|>")

    @property
    @lru_cache()
    def sot_lm(self) -> int:
        return self._get_single_token_id("<|startoflm|>")

    @property
    @lru_cache()
    def sot_prev(self) -> int:
        return self._get_single_token_id("<|startofprev|>")

    @property
    @lru_cache()
    def no_speech(self) -> int:
        return self._get_single_token_id("<|nospeech|>")

    @property
    @lru_cache()
    def no_timestamps(self) -> int:
        return self._get_single_token_id("<|notimestamps|>")

    @property
    @lru_cache()
    def timestamp_begin(self) -> int:
        return self.tokenizer.all_special_ids[-1] + 1

    @property
    @lru_cache()
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError(f"This tokenizer does not have language token configured")

        additional_tokens = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
            )
        )
        candidate = f"<|{self.language}|>"
        if candidate in additional_tokens:
            return additional_tokens[candidate]

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @property
    @lru_cache()
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @property
    @lru_cache()
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @property
    @lru_cache()
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @property
    @lru_cache()
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.tokenizer.encode(" -")[0], self.tokenizer.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer.encode(symbol), self.tokenizer.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def _get_single_token_id(self, text) -> int:
        tokens = self.tokenizer.encode(text)
        assert len(tokens) == 1, f"{text} is not encoded as a single token"
        return tokens[0]


@lru_cache(maxsize=None)
def build_tokenizer(name: str = "gpt2"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path = os.path.join(os.path.dirname(__file__), "assets", name)
    tokenizer = GPT2TokenizerFast.from_pretrained(path)

    specials = [
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]

    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
    return tokenizer


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    language: Optional[str] = None,
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        tokenizer_name = "multilingual"
        task = task or "transcribe"
        language = language or "en"
    else:
        tokenizer_name = "gpt2"
        task = None
        language = None

    tokenizer = build_tokenizer(name=tokenizer_name)
    all_special_ids: List[int] = tokenizer.all_special_ids
    sot: int = all_special_ids[1]
    translate: int = all_special_ids[-6]
    transcribe: int = all_special_ids[-5]

    langs = tuple(LANGUAGES.keys())
    sot_sequence = [sot]
    if language is not None:
        sot_sequence.append(sot + 1 + langs.index(language))
    if task is not None:
        sot_sequence.append(transcribe if task == "transcribe" else translate)

    return Tokenizer(tokenizer=tokenizer, language=language, sot_sequence=tuple(sot_sequence))
