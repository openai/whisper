from types import SimpleNamespace

import torch

from whisper.decoding import ApplyTimestampRules, DecodingOptions, DecodingTask


def test_zero_max_initial_timestamp():
    model = SimpleNamespace(
        is_multilingual=False,
        num_languages=99,
        dims=SimpleNamespace(n_text_ctx=448, n_audio_ctx=1500),
        decoder=SimpleNamespace(blocks=[]),
    )
    task = DecodingTask(model, DecodingOptions(max_initial_timestamp=0.0))
    timestamp_rules = next(
        rule for rule in task.logit_filters if isinstance(rule, ApplyTimestampRules)
    )

    logits = torch.zeros(1, task.tokenizer.encoding.n_vocab)
    tokens = torch.tensor([task.initial_tokens])
    timestamp_rules.apply(logits, tokens)

    assert torch.isfinite(logits[0, task.tokenizer.timestamp_begin])
    assert torch.isneginf(logits[0, task.tokenizer.timestamp_begin + 1 :]).all()
