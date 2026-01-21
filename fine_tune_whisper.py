#!/usr/bin/env python3
"""
Whisper 파인튜닝 스크립트 - 119 신고 전화 데이터용
"""

import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# ============================================================================
# 1. 데이터 준비
# ============================================================================

def prepare_dataset(audio_files, transcripts):
    """
    119 신고 전화 데이터를 준비합니다.

    Parameters:
    - audio_files: 오디오 파일 경로 리스트 (예: ["audio1.wav", "audio2.wav", ...])
    - transcripts: 텍스트 전사 리스트 (예: ["119입니다", "화재 신고 접수", ...])

    Returns:
    - Dataset 객체
    """
    data = {
        "audio": audio_files,
        "sentence": transcripts
    }

    dataset = Dataset.from_dict(data)
    # 오디오 파일을 자동으로 로드하고 리샘플링
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

# 예제: 실제 데이터로 교체하세요
# audio_files = ["data/119_call_001.wav", "data/119_call_002.wav", ...]
# transcripts = ["119입니다 무엇을 도와드릴까요", "화재 신고 접수합니다", ...]
# train_dataset = prepare_dataset(audio_files, transcripts)


# ============================================================================
# 2. 모델 및 프로세서 로드
# ============================================================================

def load_whisper_model(model_name="openai/whisper-small", language="Korean"):
    """
    Whisper 모델과 프로세서를 로드합니다.

    Parameters:
    - model_name: Whisper 모델 크기 (tiny, base, small, medium, large)
    - language: 대상 언어
    """
    # Feature extractor: 오디오를 mel spectrogram으로 변환
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # Tokenizer: 텍스트를 토큰으로 변환
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=language,
        task="transcribe"
    )

    # Processor: feature extractor + tokenizer
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task="transcribe"
    )

    # 모델 로드
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # 언어와 태스크 강제 설정
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = language.lower()
    model.generation_config.task = "transcribe"

    return model, processor, tokenizer


# ============================================================================
# 3. 데이터 전처리
# ============================================================================

def prepare_data(batch, processor):
    """
    배치 데이터를 전처리합니다.
    """
    # 오디오를 mel spectrogram으로 변환
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # 텍스트를 토큰으로 변환
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

    return batch


# ============================================================================
# 4. Data Collator (배치 생성)
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    음성-텍스트 시퀀스를 위한 데이터 콜레이터
    """
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # input features를 배치로 합침
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # labels를 배치로 합침
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # padding 토큰을 -100으로 변경 (loss 계산 시 무시)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # bos token 제거
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ============================================================================
# 5. 평가 메트릭 (Word Error Rate)
# ============================================================================

def compute_metrics(pred, processor, tokenizer):
    """
    Word Error Rate (WER)을 계산합니다.
    """
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # padding을 제거
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 토큰을 텍스트로 변환
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # WER 계산
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ============================================================================
# 6. 파인튜닝 실행
# ============================================================================

def fine_tune_whisper(
    train_dataset,
    eval_dataset=None,
    model_name="openai/whisper-small",
    output_dir="./whisper-finetuned-119",
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    save_steps=1000,
):
    """
    Whisper 모델을 파인튜닝합니다.

    Parameters:
    - train_dataset: 학습 데이터셋
    - eval_dataset: 검증 데이터셋 (선택)
    - model_name: 사용할 Whisper 모델
    - output_dir: 모델 저장 경로
    - num_epochs: 학습 에폭 수
    - batch_size: 배치 크기
    - learning_rate: 학습률
    - warmup_steps: Warmup 스텝 수
    - save_steps: 체크포인트 저장 주기
    """

    # 모델 로드
    model, processor, tokenizer = load_whisper_model(model_name)

    # 데이터 전처리
    train_dataset = train_dataset.map(
        lambda batch: prepare_data(batch, processor),
        remove_columns=train_dataset.column_names
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda batch: prepare_data(batch, processor),
            remove_columns=eval_dataset.column_names
        )

    # Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,  # GPU 메모리가 부족하면 증가
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,  # 메모리 절약
        fp16=True,  # Mixed precision 학습
        evaluation_strategy="steps" if eval_dataset else "no",
        per_device_eval_batch_size=batch_size if eval_dataset else 8,
        predict_with_generate=True if eval_dataset else False,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=save_steps if eval_dataset else None,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        push_to_hub=False,
    )

    # Trainer 초기화
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, tokenizer) if eval_dataset else None,
        tokenizer=processor.feature_extractor,
    )

    # 학습 시작
    print("🚀 파인튜닝을 시작합니다...")
    trainer.train()

    # 모델 저장
    print(f"💾 모델을 {output_dir}에 저장합니다...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("✅ 파인튜닝 완료!")

    return trainer


# ============================================================================
# 7. 사용 예제
# ============================================================================

if __name__ == "__main__":
    """
    실제 사용 시:
    1. 119 신고 전화 오디오 파일과 전사 텍스트를 준비
    2. prepare_dataset() 함수로 데이터셋 생성
    3. fine_tune_whisper() 함수로 파인튜닝 실행
    """

    # 예제 데이터 (실제 데이터로 교체 필요)
    example_audio_files = [
        "data/119_call_001.wav",
        "data/119_call_002.wav",
        "data/119_call_003.wav",
    ]

    example_transcripts = [
        "일일구입니다 무엇을 도와드릴까요",
        "화재 신고 접수합니다 위치를 말씀해주세요",
        "구급차가 출동하겠습니다",
    ]

    print("=" * 60)
    print("Whisper 파인튜닝 스크립트 - 119 신고 전화")
    print("=" * 60)
    print()
    print("📝 사용 방법:")
    print("1. 오디오 파일과 텍스트 전사를 준비합니다")
    print("2. prepare_dataset() 함수로 데이터셋을 생성합니다")
    print("3. fine_tune_whisper() 함수를 호출하여 학습을 시작합니다")
    print()
    print("💡 데이터 형식:")
    print("- 오디오: WAV, MP3, FLAC 등 (16kHz 권장)")
    print("- 텍스트: 정확한 한글 전사")
    print()
    print("⚙️  권장 모델 크기:")
    print("- 빠른 테스트: openai/whisper-tiny")
    print("- 균형잡힌 선택: openai/whisper-small (추천)")
    print("- 높은 정확도: openai/whisper-medium")
    print()

    # 실제 학습을 원하면 아래 주석을 해제하고 실행
    # train_dataset = prepare_dataset(example_audio_files, example_transcripts)
    # trainer = fine_tune_whisper(
    #     train_dataset=train_dataset,
    #     model_name="openai/whisper-small",
    #     output_dir="./whisper-finetuned-119",
    #     num_epochs=10,
    #     batch_size=4,
    # )
