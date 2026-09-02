import os

import numpy as np
import pytest
import torch

import whisper
from whisper.tokenizer import get_tokenizer


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


def test_carry_initial_prompt_with_full_prompt_budget():
    """A full-budget initial_prompt must be carried intact into later windows.

    Regression test for `remaining_prompt_length == 0`: `tokens[-0:]` returns
    the whole list, so the previous window's text used to be appended even
    though the initial_prompt already fills the decoder prompt budget.
    """
    tokenizer = get_tokenizer(multilingual=False)
    n_text_ctx = 448
    prompt_budget = n_text_ctx // 2 - 1
    initial_prompt = " ".join(["hello"] * prompt_budget)
    initial_prompt_tokens = tokenizer.encode(" " + initial_prompt)
    assert len(initial_prompt_tokens) == prompt_budget

    window_text_tokens = tokenizer.encode(" some speech")
    prompts = []

    class FakeModel:
        device = torch.device("cpu")
        is_multilingual = False
        num_languages = 99
        dims = whisper.ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1,
            n_audio_head=1,
            n_audio_layer=1,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=n_text_ctx,
            n_text_state=1,
            n_text_head=1,
            n_text_layer=1,
        )

        def decode(self, mel, options):
            prompts.append(list(options.prompt))
            return whisper.DecodingResult(
                audio_features=mel,
                language="en",
                tokens=[tokenizer.timestamp_begin]
                + window_text_tokens
                + [tokenizer.timestamp_begin + 1500],
                temperature=0.0,
                avg_logprob=0.0,
                no_speech_prob=0.0,
                compression_ratio=1.0,
            )

    # two 30-second windows of silence
    audio = np.zeros(2 * whisper.audio.N_SAMPLES, dtype=np.float32)
    whisper.transcribe(
        FakeModel(),
        audio,
        initial_prompt=initial_prompt,
        carry_initial_prompt=True,
        fp16=False,
    )

    assert len(prompts) == 2
    assert prompts[0] == initial_prompt_tokens
    # the second window keeps the initial prompt and carries no previous text
    assert prompts[1] == initial_prompt_tokens
