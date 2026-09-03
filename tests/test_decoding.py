import pytest
import torch

from whisper.decoding import DecodingOptions
from whisper.model import ModelDimensions, Whisper


@pytest.fixture
def model() -> Whisper:
    torch.manual_seed(0)
    dimensions = ModelDimensions(
        n_mels=80,
        n_audio_ctx=2,
        n_audio_state=4,
        n_audio_head=1,
        n_audio_layer=1,
        n_vocab=51864,
        n_text_ctx=8,
        n_text_state=4,
        n_text_head=1,
        n_text_layer=1,
    )
    model = Whisper(dimensions).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    return model


@pytest.fixture
def audio_features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 2, 4)


@pytest.mark.parametrize(
    "options",
    [
        DecodingOptions(
            language="en",
            beam_size=2,
            sample_len=1,
            without_timestamps=True,
            fp16=False,
        ),
        DecodingOptions(
            language="en",
            temperature=1.0,
            best_of=2,
            sample_len=1,
            without_timestamps=True,
            fp16=False,
        ),
    ],
    ids=["beam-search", "best-of"],
)
def test_grouped_decoding_accepts_multiple_audio_inputs(
    model: Whisper, audio_features: torch.Tensor, options: DecodingOptions
) -> None:
    torch.manual_seed(2)

    results = model.decode(audio_features, options)

    assert isinstance(results, list)
    assert len(results) == len(audio_features)
    for result, expected_features in zip(results, audio_features):
        torch.testing.assert_close(result.audio_features, expected_features)


def test_batched_beam_search_matches_independent_decodes(
    model: Whisper, audio_features: torch.Tensor
) -> None:
    options = DecodingOptions(
        language="en",
        beam_size=2,
        sample_len=1,
        without_timestamps=True,
        fp16=False,
    )

    batched = model.decode(audio_features, options)
    independent = [model.decode(features, options) for features in audio_features]

    assert isinstance(batched, list)
    assert [result.tokens for result in batched] == [
        result.tokens for result in independent
    ]


def test_single_audio_beam_search_does_not_duplicate_audio_features(
    model: Whisper, audio_features: torch.Tensor
) -> None:
    observed_batch_sizes = []
    key_projection = model.decoder.blocks[0].cross_attn.key
    handle = key_projection.register_forward_pre_hook(
        lambda _module, inputs: observed_batch_sizes.append(inputs[0].shape[0])
    )
    options = DecodingOptions(
        language="en",
        beam_size=2,
        sample_len=1,
        without_timestamps=True,
        fp16=False,
    )

    try:
        model.decode(audio_features[0], options)
    finally:
        handle.remove()

    assert observed_batch_sizes == [1]
