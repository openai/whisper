import json
import os
import wave

import numpy as np
import pytest
import torch

import whisper
from whisper import precision as precision_module
from whisper.audio import SAMPLE_RATE
from whisper.lora import load_lora
from whisper.precision import resolve_precision
from whisper.train import (
    TokenizerCache,
    build_parser,
    collate,
    edit_distance,
    error_rates,
    read_manifest,
    train,
)

WORDS = ["one", "two", "three", "four"]


def write_wav(path: str, frequency: float, seconds: float = 1.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    audio = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())


@pytest.fixture
def manifest(tmp_path):
    os.makedirs(tmp_path / "clips")
    with open(tmp_path / "train.jsonl", "w") as f:
        for i, word in enumerate(WORDS):
            write_wav(str(tmp_path / "clips" / f"{i}.wav"), 220 * (i + 1))
            f.write(json.dumps({"audio": f"clips/{i}.wav", "text": word}) + "\n")
    return str(tmp_path / "train.jsonl")


def run_training(manifest, output_dir, *extra):
    args = build_parser().parse_args(
        [
            "--model", "tiny",
            "--train_manifest", manifest,
            "--val_manifest", manifest,
            "--output_dir", str(output_dir),
            "--language", "en",
            "--batch_size", "2",
            "--lr", "1e-2",
            "--warmup_steps", "0",
            "--num_workers", "0",
            "--log_every", "1",
            *extra,
        ]
    )  # fmt: skip
    return train(args)


def test_read_manifest(tmp_path, manifest):
    samples = read_manifest(manifest)
    assert len(samples) == len(WORDS)
    assert samples[0]["audio"] == str(tmp_path / "clips" / "0.wav")
    assert "language" not in samples[0]

    csv_path = tmp_path / "train.csv"
    csv_path.write_text('audio,text,language\nclips/0.wav,"one, two",en\n')
    samples = read_manifest(str(csv_path))
    assert samples == [
        {
            "audio": str(tmp_path / "clips" / "0.wav"),
            "text": "one, two",
            "language": "en",
        }
    ]


@pytest.mark.parametrize("text", ["hello world", "আর কিছু", "नमस्ते", "你好"])
def test_first_target_token_is_never_blank(text):
    model = whisper.load_model("tiny", device="cpu")
    tokenizers = TokenizerCache(model, "en", "transcribe")
    tokens, prompt_length = tokenizers.encode(dict(text=text))
    tokenizer = tokenizers.get()
    assert tokens[prompt_length] not in tokenizer.encode(" ")
    assert tokenizer.decode(tokens[prompt_length:-1]).strip() == text


def test_collate_masks_prompt_and_padding():
    eot = 50257
    items = [
        dict(mel=torch.zeros(80, 3000), tokens=[1, 2, 3, 10, 11, eot], prompt_length=3, text="a", language="en"),
        dict(mel=torch.zeros(80, 3000), tokens=[1, 2, 3, 12, eot], prompt_length=3, text="b", language="en"),
        None,
    ]  # fmt: skip
    batch = collate(items, pad_token=eot)
    assert batch["inputs"].tolist() == [[1, 2, 3, 10, 11], [1, 2, 3, 12, eot]]
    assert batch["labels"].tolist() == [
        [-100, -100, 10, 11, eot],
        [-100, -100, 12, eot, -100],
    ]
    assert collate([None], pad_token=eot) is None


def test_error_rates():
    assert edit_distance("abc", "abd") == 1
    rates = error_rates(["Hello, world!"], ["hello world"], ["en"])
    assert rates == dict(wer=0.0, cer=0.0)
    # combining vowel signs must not be stripped or turned into word breaks
    rates = error_rates(["नमस्ते दुनिया"], ["नमस्ते दुनिय"], ["hi"])
    assert rates["wer"] == 0.5


def test_training_reduces_loss(manifest, tmp_path):
    results = run_training(manifest, tmp_path / "run", "--max_steps", "12", "--precision", "fp32", "--device", "cpu")  # fmt: skip
    assert results["precision"] == "fp32"
    assert results["final"]["loss"] < results["baseline"]["loss"]
    for name in ("adapter_best.pt", "adapter_final.pt", "run_config.json", "metrics.jsonl", "results.json"):  # fmt: skip
        assert os.path.isfile(tmp_path / "run" / name)
    with open(tmp_path / "run" / "metrics.jsonl") as f:
        losses = [r["loss"] for r in map(json.loads, f) if r["type"] == "train"]
    assert all(np.isfinite(losses))


def test_nonfinite_loss_is_skipped(manifest, tmp_path, monkeypatch):
    from whisper import train as train_module

    compute_loss = train_module.compute_loss
    calls = []

    def flaky_loss(*args, **kwargs):
        calls.append(None)
        loss = compute_loss(*args, **kwargs)
        return loss * float("nan") if len(calls) == 2 else loss

    monkeypatch.setattr(train_module, "compute_loss", flaky_loss)
    run_training(manifest, tmp_path / "run", "--max_steps", "3", "--precision", "fp32", "--device", "cpu", "--skip_baseline")  # fmt: skip
    adapter = torch.load(tmp_path / "run" / "adapter_final.pt", weights_only=True)
    assert all(torch.isfinite(v).all() for v in adapter["state_dict"].values())


def test_resume_matches_uninterrupted_run(manifest, tmp_path):
    common = ["--max_steps", "4", "--precision", "fp32", "--device", "cpu", "--skip_baseline"]  # fmt: skip
    run_training(manifest, tmp_path / "full", *common)
    run_training(manifest, tmp_path / "part", *common, "--save_every", "2")
    os.remove(tmp_path / "part" / "checkpoints" / "step_4.pt")
    run_training(manifest, tmp_path / "part", *common, "--resume", "latest")

    full = torch.load(tmp_path / "full" / "adapter_final.pt", weights_only=True)
    part = torch.load(tmp_path / "part" / "adapter_final.pt", weights_only=True)
    for name, value in full["state_dict"].items():
        assert torch.allclose(value, part["state_dict"][name], atol=1e-6), name


def test_precision_auto(monkeypatch):
    assert resolve_precision("auto", "cpu").name == "fp32"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    assert resolve_precision("auto", "cuda").name == "bf16"
    assert not resolve_precision("bf16", "cuda").use_grad_scaler
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)
    auto = resolve_precision("auto", "cuda")
    assert auto.name == "fp16" and auto.use_grad_scaler
    assert resolve_precision("fp32", "cuda").name == "fp32"


def test_precision_bf16_unsupported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "Tesla T4")
    monkeypatch.setattr(precision_module, "bf16_supported", lambda device_type: False)
    with pytest.raises(ValueError, match="--precision fp16"):
        resolve_precision("bf16", "cuda")
    with pytest.raises(ValueError):
        resolve_precision("fp16", "cpu")
    with pytest.raises(ValueError):
        resolve_precision("fp8", "cpu")


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_training_on_gpu(manifest, tmp_path):
    results = run_training(manifest, tmp_path / "run", "--max_steps", "8", "--precision", "fp16", "--device", "cuda", "--gradient_checkpointing")  # fmt: skip
    assert results["precision"] == "fp16"
    assert np.isfinite(results["final"]["loss"])
    assert results["final"]["loss"] < results["baseline"]["loss"]
    assert results["peak_memory_gb"] > 0

    model = whisper.load_model("tiny", device="cuda")
    load_lora(model, str(tmp_path / "run" / "adapter_best.pt"))
    assert all(
        p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n
    )
