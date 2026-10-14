"""
Fine-tune Whisper with LoRA adapters (default) or by updating all of its weights.

Usage:
    whisper-train --model small --train_manifest train.jsonl --val_manifest val.jsonl \
        --language en --output_dir runs/small-lora

    # full fine-tuning; saves regular Whisper checkpoints instead of adapters
    whisper-train --finetune full --model small --train_manifest train.jsonl \
        --val_manifest val.jsonl --language en --output_dir runs/small-full

Each line of a JSONL manifest describes one sample:
    {"audio": "clips/0001.wav", "text": "the transcript", "language": "en"}
"language" is optional and defaults to --language. A CSV file with "audio" and
"text" columns (and optionally "language") is also accepted. Relative audio paths
are resolved against the directory containing the manifest.
"""

import argparse
import contextlib
import csv
import functools
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import asdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions
from .lora import ALL_PARTS, ALL_TARGETS, apply_lora, load_lora, merge_lora, save_lora
from .model import Whisper
from .normalizers import EnglishTextNormalizer
from .precision import PRECISIONS, Precision, resolve_precision
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, Tokenizer, get_tokenizer

IGNORE_INDEX = -100
FINETUNE_MODES = ("lora", "full")
DEFAULT_LR = {"lora": 5e-4, "full": 1e-5}


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_manifest(path: str) -> List[Dict[str, str]]:
    """
    Read a JSONL or CSV manifest. Returns a list of dicts with the keys "audio",
    "text" and, if present, "language", with audio paths resolved relative to the
    directory of the manifest.
    """
    root = os.path.dirname(os.path.abspath(path))

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and not {"audio", "text"} <= set(rows[0].keys()):
            raise ValueError(f"{path}: CSV manifests need 'audio' and 'text' columns")
    else:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {e}")
                if not isinstance(row, dict) or "audio" not in row or "text" not in row:
                    raise ValueError(
                        f"{path}:{line_number}: each line needs 'audio' and 'text'"
                    )
                rows.append(row)

    samples = []
    for row in rows:
        sample = {
            "audio": os.path.join(root, os.path.expanduser(str(row["audio"]))),
            "text": "" if row["text"] is None else str(row["text"]),
        }
        if row.get("language"):
            sample["language"] = str(row["language"])
        samples.append(sample)
    return samples


class TokenizerCache:
    """Builds the token sequences for each sample, with one tokenizer per language"""

    def __init__(self, model: Whisper, language: Optional[str], task: str):
        self.multilingual = model.is_multilingual
        self.num_languages = model.num_languages
        self.language = language
        self.task = task
        self.tokenizers: Dict[Optional[str], Tokenizer] = {}

    def get(self, language: Optional[str] = None) -> Tokenizer:
        if not self.multilingual:
            language = None
        else:
            language = language or self.language
            if language is None:
                raise ValueError(
                    "A language is required for multilingual models: set --language "
                    "or add a 'language' field to every sample in the manifest"
                )
            language = TO_LANGUAGE_CODE.get(language.lower(), language.lower())
        if language not in self.tokenizers:
            self.tokenizers[language] = get_tokenizer(
                self.multilingual,
                num_languages=self.num_languages,
                language=language,
                task=self.task,
            )
        return self.tokenizers[language]

    def encode(self, sample: Dict[str, str]):
        """Returns (tokens, prompt_length) for a sample"""
        tokenizer = self.get(sample.get("language"))
        prompt = list(tokenizer.sot_sequence_including_notimestamps)
        text = tokenizer.encode(" " + sample["text"].strip())
        if text[:1] == tokenizer.encode(" "):
            # In some scripts (e.g. Bengali) the leading space is not merged into
            # the first word and becomes a lone space token, which the decoder
            # never samples first (see SuppressBlank in decoding.py); drop it.
            text = text[1:]
        return prompt + text + [tokenizer.eot], len(prompt)


class SpeechDataset(Dataset):
    def __init__(
        self, samples: List[Dict[str, str]], tokenizers: TokenizerCache, n_mels: int
    ):
        self.samples = samples
        self.tokenizers = tokenizers
        self.n_mels = n_mels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            audio = load_audio(sample["audio"])
        except Exception as e:
            warn(f"skipping {sample['audio']}: failed to load audio ({e})")
            return None
        if len(audio) > N_SAMPLES:
            warn(
                f"skipping {sample['audio']}: {len(audio) / SAMPLE_RATE:.1f}s is longer "
                f"than the 30s that Whisper can process in one window"
            )
            return None

        mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=self.n_mels)
        tokens, prompt_length = self.tokenizers.encode(sample)
        return dict(
            mel=mel,
            tokens=tokens,
            prompt_length=prompt_length,
            text=sample["text"],
            language=self.tokenizers.get(sample.get("language")).language,
        )


def collate(items: List[Optional[dict]], pad_token: int) -> Optional[dict]:
    items = [item for item in items if item is not None]
    if not items:
        return None

    max_length = max(len(item["tokens"]) for item in items) - 1
    inputs = torch.full((len(items), max_length), pad_token, dtype=torch.long)
    labels = torch.full((len(items), max_length), IGNORE_INDEX, dtype=torch.long)
    for i, item in enumerate(items):
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        inputs[i, : len(tokens) - 1] = tokens[:-1]
        labels[i, : len(tokens) - 1] = tokens[1:]
        # don't train on predicting the start-of-transcript prompt; the first
        # supervised position is the one predicting the first text token
        labels[i, : item["prompt_length"] - 1] = IGNORE_INDEX

    return dict(
        mel=torch.stack([item["mel"] for item in items]),
        inputs=inputs,
        labels=labels,
        texts=[item["text"] for item in items],
        languages=[item["language"] for item in items],
    )


def filter_samples(
    samples: List[Dict[str, str]], tokenizers: TokenizerCache, n_text_ctx: int
) -> List[Dict[str, str]]:
    """Drop samples whose audio file is missing or whose transcript is too long"""
    kept, missing, too_long = [], [], []
    for sample in samples:
        if not os.path.isfile(sample["audio"]):
            missing.append(sample["audio"])
            continue
        tokens, _ = tokenizers.encode(sample)
        if len(tokens) > n_text_ctx:
            too_long.append(sample["audio"])
            continue
        kept.append(sample)

    for paths, reason in [
        (missing, "audio file not found"),
        (too_long, f"transcript longer than the decoder context ({n_text_ctx} tokens)"),
    ]:
        if paths:
            examples = ", ".join(paths[:3]) + (", ..." if len(paths) > 3 else "")
            warn(f"skipping {len(paths)} sample(s): {reason}: {examples}")
    return kept


class EpochSampler(Sampler):
    """A seeded shuffling sampler that can start from an offset, for exact resumption"""

    def __init__(self, length: int, seed: int, shuffle: bool = True):
        self.length = length
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch = epoch
        self.start = start

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.length, generator=generator).tolist()
        else:
            order = list(range(self.length))
        return iter(order[self.start :])

    def __len__(self):
        return self.length - self.start


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def edit_distance(reference: List[str], hypothesis: List[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, 1):
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)
            )
        previous = current
    return previous[-1]


def normalize_text(text: str, language: Optional[str]) -> str:
    """
    Normalize a transcript before computing WER/CER.

    English uses Whisper's EnglishTextNormalizer. Other languages only have
    punctuation and symbols removed and are lowercased; combining marks are kept,
    since in many scripts (e.g. Devanagari, Bengali, Tamil, Thai) they are vowel
    signs that are part of the word rather than optional diacritics.
    """
    if language in (None, "en"):
        return EnglishTextNormalizer()(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def error_rates(
    references: List[str], hypotheses: List[str], languages: List[Optional[str]]
) -> Dict[str, float]:
    word_errors = word_total = char_errors = char_total = 0
    for reference, hypothesis, language in zip(references, hypotheses, languages):
        reference = normalize_text(reference, language)
        hypothesis = normalize_text(hypothesis, language)
        word_errors += edit_distance(reference.split(), hypothesis.split())
        word_total += len(reference.split())
        char_errors += edit_distance(list(reference), list(hypothesis))
        char_total += len(reference)
    return dict(
        wer=word_errors / max(word_total, 1), cer=char_errors / max(char_total, 1)
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def enable_gradient_checkpointing(model: Whisper) -> None:
    """Recompute each residual block's activations in the backward pass to save memory"""

    def wrap(forward):
        @functools.wraps(forward)
        def checkpointed(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)

        return checkpointed

    for block in [*model.encoder.blocks, *model.decoder.blocks]:
        block.forward = wrap(block.forward)


def lr_lambda(step: int, warmup: int, total: int, schedule: str) -> float:
    if step < warmup:
        return (step + 1) / (warmup + 1)
    if schedule == "constant":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(progress, 1.0)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0 - progress


def compute_loss(
    model: Whisper, batch: dict, precision: Precision, train_encoder: bool
) -> torch.Tensor:
    device = model.device
    mel = batch["mel"].to(device, non_blocking=True)
    inputs = batch["inputs"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    with precision.autocast():
        if train_encoder:
            audio_features = model.encoder(mel)
        else:
            with torch.no_grad():
                audio_features = model.encoder(mel)
        logits = model.decoder(inputs, audio_features)

    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.no_grad()
def evaluate(
    model: Whisper,
    loader: DataLoader,
    precision: Precision,
    task: str,
    beam_size: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_batches = 0.0, 0
    references, hypotheses, languages = [], [], []

    for batch in loader:
        if batch is None:
            continue
        total_loss += compute_loss(model, batch, precision, True).item()
        total_batches += 1

        mel = batch["mel"].to(model.device)
        for language in dict.fromkeys(batch["languages"]):
            index = [i for i, lang in enumerate(batch["languages"]) if lang == language]
            options = DecodingOptions(
                task=task,
                language=language,
                without_timestamps=True,
                beam_size=beam_size,
                fp16=precision.name == "fp16",
            )
            # fp16 decoding uses Whisper's own fp16 inference path; bf16 decoding
            # runs the fp32 path under autocast
            if precision.name == "bf16":
                autocast = precision.autocast()
            else:
                autocast = contextlib.nullcontext()
            with autocast:
                results = model.decode(mel[index], options)
            hypotheses.extend(result.text for result in results)
            references.extend(batch["texts"][i] for i in index)
            languages.extend([language] * len(index))

    model.train()
    metrics = dict(loss=total_loss / max(total_batches, 1))
    metrics.update(error_rates(references, hypotheses, languages))
    metrics["samples"] = len(references)
    metrics["predictions"] = [
        dict(reference=r, hypothesis=h, language=lang)
        for r, h, lang in zip(references, hypotheses, languages)
    ]
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_parameters(model: Whisper):
    return [p for p in model.parameters() if p.requires_grad]


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, state: dict):
    trainable = {
        n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
    }
    torch.save(
        dict(
            model=trainable,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            state=state,
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=(
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            ),
        ),
        path,
    )


def load_checkpoint(path: str, model, optimizer, scheduler, scaler) -> dict:
    # resumable checkpoints contain the optimizer and RNG states, not just tensors
    kwargs = {"weights_only": False} if torch.__version__ >= "1.13" else {}
    checkpoint = torch.load(path, map_location="cpu", **kwargs)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise ValueError(
            f"checkpoint {path} does not match the model: {unexpected[:5]}"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint["state"]


def save_model(model: Whisper, path: str) -> None:
    """Save the whole model as a regular Whisper checkpoint, loadable by load_model()"""
    # stored in fp16 like the official checkpoints; load_model() casts it back
    state_dict = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
    torch.save({"dims": asdict(model.dims), "model_state_dict": state_dict}, path)


def list_checkpoints(output_dir: str) -> List[str]:
    """Paths of the resumable checkpoints in output_dir, oldest first"""
    directory = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(directory):
        return []
    steps = sorted(
        int(m.group(1))
        for name in os.listdir(directory)
        if (m := re.fullmatch(r"step_(\d+)\.pt", name))
    )
    return [os.path.join(directory, f"step_{step}.pt") for step in steps]


def latest_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


class MetricsLogger:
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "metrics.jsonl")

    def __call__(self, **record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, object]:
    from . import load_model

    full = args.finetune == "full"
    if not full and args.freeze_encoder:
        raise ValueError(
            "--freeze_encoder only applies to --finetune full; "
            "use --lora_apply_to decoder for LoRA"
        )
    if full and args.export_merged:
        raise ValueError("--export_merged only applies to --finetune lora")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.finetune]

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    precision = resolve_precision(args.precision, device)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    set_seed(args.seed)
    log(f"Precision: {precision.name} (requested: {args.precision}) on {device}")
    log(f"Fine-tuning: {args.finetune}, learning rate {lr:g}")

    model = load_model(args.model, device="cpu", download_root=args.model_dir)
    if not model.is_multilingual and args.language not in (None, "en", "english"):
        warn(f"{args.model} is English-only; ignoring --language {args.language}")

    tokenizers = TokenizerCache(model, args.language, args.task)
    train_samples = read_manifest(args.train_manifest)
    train_samples = filter_samples(train_samples, tokenizers, model.dims.n_text_ctx)
    val_samples = []
    if args.val_manifest:
        val_samples = read_manifest(args.val_manifest)
        val_samples = filter_samples(val_samples, tokenizers, model.dims.n_text_ctx)
        if args.max_eval_samples:
            val_samples = val_samples[: args.max_eval_samples]
    if not train_samples:
        raise ValueError("no usable training samples")
    log(f"Samples: {len(train_samples)} train, {len(val_samples)} validation")

    if full:
        train_encoder = not args.freeze_encoder
        for p in model.parameters():
            p.requires_grad = True
        if not train_encoder:
            model.encoder.requires_grad_(False)
    else:
        apply_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            apply_to=args.lora_apply_to,
        )
        train_encoder = "encoder" in model.lora_config["apply_to"]
    # the parameters stay in fp32 and act as master weights under autocast
    model.to(device)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    parameters = trainable_parameters(model)
    n_trainable = sum(p.numel() for p in parameters)
    n_total = sum(p.numel() for p in model.parameters())
    log(
        f"Trainable parameters: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.2f}%)"
    )

    pad_token = tokenizers.get().eot
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=functools.partial(collate, pad_token=pad_token),
        pin_memory=device.type == "cuda",
        # a separate generator keeps data loading from consuming the global RNG,
        # so that resumed runs reproduce the same dropout masks
        generator=torch.Generator().manual_seed(args.seed),
    )
    sampler = EpochSampler(len(train_samples), args.seed)
    train_loader = DataLoader(
        SpeechDataset(train_samples, tokenizers, model.dims.n_mels),
        sampler=sampler,
        drop_last=False,
        **loader_args,
    )
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            SpeechDataset(val_samples, tokenizers, model.dims.n_mels), **loader_args
        )

    batches_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else int(0.05 * total_steps)
    )

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        functools.partial(
            lr_lambda,
            warmup=warmup_steps,
            total=total_steps,
            schedule=args.lr_scheduler,
        ),
    )
    scaler = precision.grad_scaler()

    state = dict(step=0, epoch=0, batch_in_epoch=0, best_wer=None, best_loss=None)
    resume = args.resume
    if resume == "latest":
        resume = latest_checkpoint(args.output_dir)
        if resume is None:
            log("No checkpoint found to resume from; starting from scratch")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        log(f"Resumed from {resume} at step {state['step']}")

    run_config = dict(
        vars(args),
        lr=lr,
        precision_resolved=precision.name,
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        torch_version=torch.__version__,
        lora_config=getattr(model, "lora_config", None),
        dims=asdict(model.dims),
        trainable_parameters=n_trainable,
        total_parameters=n_total,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    metrics_log = MetricsLogger(args.output_dir)
    prefix = "model" if full else "adapter"
    best_path = os.path.join(args.output_dir, f"{prefix}_best.pt")
    final_path = os.path.join(args.output_dir, f"{prefix}_final.pt")
    results: Dict[str, object] = dict(finetune=args.finetune, precision=precision.name)

    def save_weights(path: str) -> None:
        if full:
            save_model(model, path)
        else:
            save_lora(model, path, base_model=args.model)

    def run_eval(step: int, tag: str) -> Optional[dict]:
        if val_loader is None:
            return None
        start = time.time()
        metrics = evaluate(model, val_loader, precision, args.task, args.eval_beam_size)
        log(
            f"[eval {tag}] step {step}: loss {metrics['loss']:.4f} "
            f"WER {100 * metrics['wer']:.2f}% CER {100 * metrics['cer']:.2f}% "
            f"({metrics['samples']} samples, {time.time() - start:.1f}s)"
        )
        predictions = metrics.pop("predictions")
        path = os.path.join(args.output_dir, "predictions", f"step_{step}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        metrics_log(type="eval", tag=tag, step=step, **metrics)
        return metrics

    def maybe_save_best(metrics: Optional[dict]) -> None:
        if metrics is None:
            return
        if state["best_wer"] is None or metrics["wer"] < state["best_wer"]:
            state["best_wer"] = metrics["wer"]
            state["best_loss"] = metrics["loss"]
            save_weights(best_path)
            log(f"New best WER {100 * metrics['wer']:.2f}%; saved {best_path}")

    if state["step"] == 0 and not args.skip_baseline:
        # nothing has been trained yet (LoRA B matrices start at zero), so this is
        # the unmodified base model
        results["baseline"] = run_eval(0, "baseline")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    nonfinite_streak = 0
    window_loss, window_batches, window_steps = 0.0, 0, 0
    step_times: List[float] = []
    micro_batch = 0
    step_start = time.time()

    while state["step"] < total_steps:
        sampler.set_epoch(state["epoch"], state["batch_in_epoch"] * args.batch_size)
        for batch in train_loader:
            state["batch_in_epoch"] += 1
            if batch is None:
                continue

            if micro_batch % args.grad_accum == 0:
                step_start = time.time()
            loss = compute_loss(model, batch, precision, train_encoder)
            if not torch.isfinite(loss):
                nonfinite_streak += 1
                warn(f"non-finite loss {loss.item()} at step {state['step']}")
                if nonfinite_streak >= 10:
                    raise RuntimeError(
                        "loss was NaN/inf for 10 consecutive batches; try a lower "
                        "learning rate or --precision fp32/bf16"
                    )
                continue  # skip the batch rather than propagating NaN/inf
            nonfinite_streak = 0
            window_loss += loss.item()
            window_batches += 1
            scaler.scale(loss / args.grad_accum).backward()
            micro_batch += 1

            if micro_batch % args.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not precision.use_grad_scaler and not torch.isfinite(grad_norm):
                # GradScaler skips such steps in fp16; do the same for fp32/bf16
                warn(f"non-finite gradient norm at step {state['step']}; skipping")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["step"] += 1
            step_times.append(time.time() - step_start)
            window_steps += 1

            step = state["step"]
            if step % args.log_every == 0 or step == total_steps:
                record = dict(
                    type="train",
                    step=step,
                    epoch=state["epoch"],
                    loss=window_loss / max(window_batches, 1),
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=float(grad_norm),
                    seconds_per_step=float(np.mean(step_times[-window_steps:])),
                )
                if precision.use_grad_scaler:
                    record["loss_scale"] = scaler.get_scale()
                if device.type == "cuda":
                    record["peak_memory_gb"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                log(
                    " ".join(
                        f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in record.items()
                    )
                )
                metrics_log(**record)
                window_loss, window_batches, window_steps = 0.0, 0, 0

            if args.eval_every and step % args.eval_every == 0 and step < total_steps:
                maybe_save_best(run_eval(step, "step"))
            if args.save_every and step % args.save_every == 0:
                path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
                save_checkpoint(path, model, optimizer, scheduler, scaler, state)
                log(f"Saved checkpoint {path}")
                if args.keep_checkpoints:
                    for old in list_checkpoints(args.output_dir)[
                        : -args.keep_checkpoints
                    ]:
                        os.remove(old)
            if step >= total_steps:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0
            if not args.eval_every:
                maybe_save_best(run_eval(state["step"], f"epoch {state['epoch']}"))
            continue
        break

    save_weights(final_path)
    log(f"Saved final {prefix} to {final_path}")
    final = run_eval(state["step"], "final")
    maybe_save_best(final)

    results.update(
        final=final,
        best_wer=state["best_wer"],
        steps=state["step"],
        trainable_parameters=n_trainable,
        trainable_percent=100 * n_trainable / n_total,
        seconds_per_step=float(np.median(step_times)) if step_times else None,
    )
    if device.type == "cuda":
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    if results.get("baseline") and final:
        log(
            f"Baseline WER {100 * results['baseline']['wer']:.2f}% -> "
            f"fine-tuned WER {100 * final['wer']:.2f}% "
            f"(best {100 * state['best_wer']:.2f}%)"
        )

    if args.export_merged:
        adapter = best_path if os.path.isfile(best_path) else final_path
        merged = load_model(args.model, device="cpu", download_root=args.model_dir)
        merge_lora(load_lora(merged, adapter, base_model=args.model))
        merged_path = os.path.join(args.output_dir, "merged_model.pt")
        save_model(merged, merged_path)
        log(f"Exported merged model ({adapter}) to {merged_path}")
        results["merged_model"] = merged_path

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    from . import available_models

    def valid_model_name(name):
        if name in available_models() or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {available_models()} or path to a model checkpoint"
        )

    def valid_language(name):
        if name.lower() in LANGUAGES or name.lower() in TO_LANGUAGE_CODE:
            return name
        raise ValueError(f"unsupported language: {name}")

    # fmt: off
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA or full fine-tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="small", type=valid_model_name, help="name of the Whisper model to fine-tune, or path to a checkpoint")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--train_manifest", type=str, required=True, help="JSONL or CSV file with the training samples")
    parser.add_argument("--val_manifest", type=str, default=None, help="JSONL or CSV file with the validation samples")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="directory to save adapters or models, checkpoints and logs")
    parser.add_argument("--language", type=valid_language, default=None, help="language of samples that don't specify one")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to train X->X transcription or X->English translation")

    parser.add_argument("--finetune", type=str, default="lora", choices=FINETUNE_MODES, help="lora: train small adapters on a frozen model; full: update all weights (needs much more memory)")
    parser.add_argument("--freeze_encoder", action="store_true", help="with --finetune full, train only the decoder")
    parser.add_argument("--lora_r", type=int, default=8, help="rank of the LoRA update")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA scaling numerator; the update is scaled by alpha / r")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="dropout applied to the input of the LoRA layers")
    parser.add_argument("--lora_targets", type=str, default="query,value", help=f"comma-separated layers to adapt, from {','.join(ALL_TARGETS)}, or 'all'")
    parser.add_argument("--lora_apply_to", type=str, default="encoder,decoder", help=f"comma-separated parts of the model to adapt, from {','.join(ALL_PARTS)}")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to train on, e.g. cuda, cuda:1 or cpu")
    parser.add_argument("--precision", type=str, default="auto", choices=PRECISIONS, help="auto: bf16 if the GPU supports it, else fp16 on GPU and fp32 on CPU")
    parser.add_argument("--epochs", type=int, default=3, help="number of passes over the training set")
    parser.add_argument("--max_steps", type=int, default=None, help="number of optimizer steps; overrides --epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="number of samples per forward pass")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of batches to accumulate per optimizer step")
    parser.add_argument("--lr", type=float, default=None, help=f"peak learning rate; defaults to {DEFAULT_LR['lora']:g} for LoRA and {DEFAULT_LR['full']:g} for full fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine", "constant"], help="learning rate decay after warmup")
    parser.add_argument("--warmup_steps", type=int, default=None, help="linear warmup steps; defaults to 5%% of the total steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping threshold")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="recompute activations in the backward pass to save memory")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading processes")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--eval_every", type=int, default=0, help="evaluate every N steps; 0 evaluates at the end of each epoch")
    parser.add_argument("--eval_beam_size", type=int, default=None, help="beam size for evaluation decoding; greedy if not set")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="evaluate on at most this many validation samples")
    parser.add_argument("--skip_baseline", action="store_true", help="don't evaluate the base model before training")
    parser.add_argument("--log_every", type=int, default=10, help="log training loss every N steps")
    parser.add_argument("--save_every", type=int, default=0, help="save a resumable checkpoint every N steps; 0 disables")
    parser.add_argument("--keep_checkpoints", type=int, default=0, help="keep only the N most recent resumable checkpoints; 0 keeps all")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from, or 'latest' to use the newest one in --output_dir")
    parser.add_argument("--export_merged", action="store_true", help="with LoRA, also save the best adapter merged into a full Whisper checkpoint")
    # fmt: on
    return parser


def cli():
    args = build_parser().parse_args()
    try:
        train(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    cli()
