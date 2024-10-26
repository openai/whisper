import os

import pytest
import torch

import whisper
from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import get_tokenizer
from whisper.transcribe import Transcriber


@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked


class MockTokenizer:
    def __init__(self, language, **kw):
        self.language, self._kw = language, kw
        for k, v in kw.items():
            setattr(self, k, v)

    def encode(self, prompt):
        return [self.language, self, prompt]


class OnDemand:
    def __init__(self, seq=(), relative=True):
        self.seq, self.relative = seq, relative
        self.prev, self.given = 0, 0

    def __getitem__(self, key):
        _key = self.given if self.relative else key
        self.prev = (
            self.seq[_key]
            if _key < len(self.seq)
            else int(input(f"lang @ {_key}: ") or self.prev)
        )
        self.given += 1
        return self.prev

    def __len__(self):
        return CHUNK_LENGTH + 2 if self.relative else len(self.seq)


class TranscriberTest(Transcriber):
    sample = object()
    dtype = torch.float32
    model = type(
        "MockModel",
        (),
        {"is_multilingual": True, "num_languages": None, "device": torch.device("cpu")},
    )()
    _seek = 0

    def __init__(self, seq=None):
        super().__init__(self.model, initial_prompt="")
        self.seq = OnDemand(seq or ())
        self.result = []
        self.latest = torch.zeros((0,))
        for i in range(len(self.seq)):
            self._seek = i
            self.frame_offset = max(0, i + 1 - CHUNK_LENGTH)
            res = self.initial_prompt_tokens
            assert res[0] == self.seq.prev
            self.result.append(res[1:])
            if seq is None:
                print(res)

    def detect_language(self, mel=None):
        self.result.append([self.sample, mel])
        return self.seq[self._seek]

    def get_tokenizer(self, multilingual, language, **kw):
        return MockTokenizer(language, **{"multilingual": multilingual, **kw})

    @property
    def rle(self):
        res = []
        for i, *j in self.result:
            if i is self.sample:
                res.append(0)
            else:
                res[-1] += 1
        return res


def test_language():
    res = TranscriberTest([0, 0, 1, 0, 0, 0, 0, 0, 0]).rle
    assert res == [1, 2, 1, 1, 2, 4, 8, 11, 2]
