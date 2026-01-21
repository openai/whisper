# Whisper 파인튜닝 가이드 - 119 신고 전화 데이터

이 가이드는 119 신고 전화 음성과 텍스트 데이터로 Whisper 모델을 파인튜닝하는 방법을 설명합니다.

## 📋 목차

1. [준비사항](#준비사항)
2. [데이터 준비](#데이터-준비)
3. [설치](#설치)
4. [파인튜닝 실행](#파인튜닝-실행)
5. [학습된 모델 사용](#학습된-모델-사용)
6. [고급 옵션](#고급-옵션)

---

## 준비사항

### 필요한 하드웨어
- **GPU**: NVIDIA GPU (VRAM 8GB 이상 권장)
  - Whisper-tiny/base: 4GB VRAM
  - Whisper-small: 8GB VRAM
  - Whisper-medium: 16GB VRAM
  - Whisper-large: 24GB VRAM

### 데이터 요구사항
- **최소**: 1시간 분량의 오디오 + 전사
- **권장**: 10시간 이상
- **최적**: 100시간 이상

---

## 데이터 준비

### 1. 데이터 구조

다음과 같이 데이터를 구성하세요:

```
data/
├── train/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── test/
│   ├── audio_test_001.wav
│   └── ...
├── train.csv
└── test.csv
```

### 2. CSV 파일 형식

`train.csv`와 `test.csv`는 다음 형식을 따릅니다:

```csv
file_path,transcription
data/train/audio_001.wav,일일구입니다 무엇을 도와드릴까요
data/train/audio_002.wav,화재 신고 접수합니다
data/train/audio_003.wav,환자의 상태를 말씀해주세요
```

### 3. 오디오 형식
- **형식**: WAV, MP3, FLAC
- **샘플레이트**: 16kHz 권장 (자동 리샘플링됨)
- **채널**: 모노 권장 (스테레오도 가능)

---

## 설치

### 1. 의존성 설치

```bash
# 파인튜닝용 패키지 설치
pip install -r requirements_finetuning.txt

# 또는 개별 설치
pip install transformers datasets accelerate evaluate
pip install librosa soundfile jiwer
pip install torch torchaudio tensorboard
```

---

## 파인튜닝 실행

### 방법 1: Python 스크립트 사용

#### A. 데이터셋 준비 스크립트

```python
import pandas as pd
from datasets import Dataset, Audio

# CSV 파일 읽기
df = pd.read_csv("data/train.csv")

# Dataset 생성
train_dataset = Dataset.from_dict({
    "audio": df["file_path"].tolist(),
    "sentence": df["transcription"].tolist()
})

# 오디오 컬럼 캐스팅 (자동 로드 및 리샘플링)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

print(f"학습 데이터: {len(train_dataset)}개")
```

#### B. 파인튜닝 실행

```python
from fine_tune_whisper import fine_tune_whisper

# 파인튜닝 시작
trainer = fine_tune_whisper(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # 선택사항
    model_name="openai/whisper-small",  # 모델 크기 선택
    output_dir="./whisper-finetuned-119",
    num_epochs=10,
    batch_size=8,  # GPU 메모리에 맞게 조정
    learning_rate=1e-5,
)
```

### 방법 2: 커맨드라인 스크립트

더 간단한 실행을 위한 스크립트:

```python
# train_119_model.py
import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from fine_tune_whisper import fine_tune_whisper

# 데이터 로드
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Dataset 생성
train_dataset = Dataset.from_dict({
    "audio": train_df["file_path"].tolist(),
    "sentence": train_df["transcription"].tolist()
}).cast_column("audio", Audio(sampling_rate=16000))

test_dataset = Dataset.from_dict({
    "audio": test_df["file_path"].tolist(),
    "sentence": test_df["transcription"].tolist()
}).cast_column("audio", Audio(sampling_rate=16000))

# 파인튜닝
trainer = fine_tune_whisper(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    model_name="openai/whisper-small",
    output_dir="./whisper-finetuned-119",
    num_epochs=10,
    batch_size=4,
)
```

실행:
```bash
python train_119_model.py
```

---

## 학습된 모델 사용

### 1. 모델 로드 및 추론

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# 파인튜닝된 모델 로드
model_path = "./whisper-finetuned-119"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model = model.to("cuda")

# 오디오 파일 로드
audio_path = "test_119_call.wav"
audio, sr = librosa.load(audio_path, sr=16000)

# 추론
input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features.to("cuda")

# 텍스트 생성
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print(f"전사 결과: {transcription}")
```

### 2. Whisper API와 동일하게 사용

```python
import whisper

# 파인튜닝된 모델을 whisper처럼 사용
model = whisper.load_model("./whisper-finetuned-119")
result = model.transcribe("test_119_call.wav")
print(result["text"])
```

---

## 고급 옵션

### 1. 하이퍼파라미터 튜닝

```python
trainer = fine_tune_whisper(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    model_name="openai/whisper-small",
    output_dir="./whisper-finetuned-119",

    # 학습 설정
    num_epochs=20,  # 에폭 증가
    batch_size=4,  # 메모리 부족 시 감소
    learning_rate=5e-6,  # 학습률 조정
    warmup_steps=1000,  # Warmup 스텝
    save_steps=500,  # 체크포인트 저장 빈도
)
```

### 2. GPU 메모리 최적화

GPU 메모리가 부족한 경우:

```python
training_args = Seq2SeqTrainingArguments(
    # ... 기존 설정 ...

    # 메모리 절약 옵션
    per_device_train_batch_size=2,  # 배치 크기 감소
    gradient_accumulation_steps=4,  # Gradient accumulation으로 보완
    gradient_checkpointing=True,  # 메모리 절약
    fp16=True,  # Mixed precision
)
```

### 3. 다양한 모델 크기 비교

| 모델 크기 | 파라미터 | VRAM | 학습 시간 | 정확도 |
|---------|---------|------|---------|--------|
| tiny | 39M | ~4GB | 빠름 | 낮음 |
| base | 74M | ~4GB | 빠름 | 중간 |
| small | 244M | ~8GB | 중간 | 높음 |
| medium | 769M | ~16GB | 느림 | 매우 높음 |

### 4. 데이터 증강

```python
# 배경 소음 추가, 속도 변경 등으로 데이터 증강
import numpy as np

def augment_audio(audio, noise_factor=0.005):
    """배경 소음 추가"""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio
```

### 5. 특수 토큰 추가 (119 전문 용어)

119 신고 전화에 자주 나오는 특수 용어를 토큰화할 수 있습니다:

```python
# 특수 용어 리스트
special_tokens = [
    "CPR", "AED", "심정지", "골든타임", "응급처치"
]

tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

---

## 📊 학습 모니터링

### TensorBoard 사용

```bash
# TensorBoard 실행
tensorboard --logdir ./whisper-finetuned-119/runs

# 브라우저에서 http://localhost:6006 접속
```

---

## 🔧 문제 해결

### 1. CUDA Out of Memory
- `batch_size` 감소
- `gradient_accumulation_steps` 증가
- 더 작은 모델 사용

### 2. 학습이 느림
- GPU 사용 확인: `torch.cuda.is_available()`
- `fp16=True` 설정 확인
- 더 작은 모델로 시작

### 3. WER이 개선되지 않음
- 더 많은 데이터 추가
- 학습 에폭 증가
- Learning rate 조정
- 데이터 품질 확인

---

## 💡 팁

1. **작은 모델로 시작**: tiny나 base로 빠르게 테스트 후 small/medium으로 확장
2. **검증 데이터 활용**: WER을 모니터링하여 과적합 방지
3. **체크포인트 저장**: 정기적으로 모델 저장하여 최적 모델 선택
4. **도메인 데이터**: 119 신고 전화와 유사한 데이터가 많을수록 좋음

---

## 📚 참고 자료

- [Hugging Face Whisper 문서](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Whisper 논문](https://arxiv.org/abs/2212.04356)
- [파인튜닝 튜토리얼](https://huggingface.co/blog/fine-tune-whisper)

---

## 🎯 예상 결과

| 데이터 크기 | 학습 시간 (small) | WER 개선 |
|-----------|-----------------|---------|
| 1시간 | ~2-3시간 | 10-20% |
| 10시간 | ~1일 | 30-50% |
| 100시간 | ~1주 | 60-80% |

*GPU: NVIDIA RTX 3090 기준

---

질문이나 문제가 있으시면 GitHub Issues에 등록해주세요!
