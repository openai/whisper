import pytest
import torch

from whisper.decoding import DecodingOptions, decode
from whisper.model import ModelDimensions, Whisper
from whisper.tokenizer import get_tokenizer


def _make_model(n_vocab: int) -> Whisper:
    torch.manual_seed(0)
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=n_vocab,
        n_text_ctx=448,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims).eval()
    for param in model.parameters():
        torch.nn.init.uniform_(param, -0.05, 0.05)
    return model


def test_decode_english_only_default_language():
    model = _make_model(n_vocab=51864)
    assert not model.is_multilingual

    mel = torch.zeros(80, 3000)
    default_result = decode(model, mel, DecodingOptions(fp16=False, sample_len=4))
    explicit_result = decode(
        model, mel, DecodingOptions(language="en", fp16=False, sample_len=4)
    )

    assert default_result.language == "en"
    assert default_result.tokens == explicit_result.tokens
    assert default_result.text == explicit_result.text

    batched_results = model.decode(torch.zeros(2, 80, 3000), fp16=False, sample_len=4)
    assert [r.language for r in batched_results] == ["en", "en"]


def test_decode_english_only_lang_id_raises():
    model = _make_model(n_vocab=51864)
    mel = torch.zeros(80, 3000)

    with pytest.raises(ValueError, match="doesn't have language tokens"):
        decode(model, mel, DecodingOptions(task="lang_id", fp16=False))


def test_decode_multilingual_default_language_detects():
    model = _make_model(n_vocab=51865)
    assert model.is_multilingual

    tokenizer = get_tokenizer(multilingual=True, num_languages=model.num_languages)
    fr_token = tokenizer.to_language_token("fr")
    with torch.no_grad():
        model.decoder.token_embedding.weight.zero_()
        model.decoder.positional_embedding.zero_()
        for block in model.decoder.blocks:
            for param in block.parameters():
                param.zero_()
        model.decoder.ln.weight.fill_(1.0)
        model.decoder.ln.bias.fill_(1.0)
        model.decoder.token_embedding.weight[fr_token].fill_(10.0)

    mel = torch.zeros(80, 3000)
    result = decode(model, mel, DecodingOptions(fp16=False, sample_len=2))
    assert result.language == "fr"
