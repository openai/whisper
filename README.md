
# Whisper <!-- omit in toc -->

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

## Contents <!-- omit in toc -->

- [What is Whisper](#what-is-whisper)
- [Setup](#setup)
  - [Prerequsites](#prerequisites)
  - [Installation](#installation)
  - [Installation troubleshooting](#installation-troubleshooting)
- [Available models and languages](#available-models-and-languages)
- [Performance](#performance)
- [Command-line usage](#command-line-usage)
- [Python usage](#python-usage)
- [More examples](#more-examples)
- [License](#license)

## What is Whisper

Whisper is a multilingual speech recognition model for general purposes, including speech translation and language identification. Whisper is trained on a large dataset of diverse audio.

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks. The tasks include multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder. As a result, a single model replaces many steps in traditional speech processing. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models. The codebase should be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation.

## Setup

### Prerequisites

* Whisper requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system. The command-line tool is available from most package managers. To install [`ffmpeg`](https://ffmpeg.org/), use one of the following commands for your operating system:

**on Ubuntu or Debian**
```bash
sudo apt update && sudo apt install ffmpeg
```

**on Arch Linux**
```bash
sudo pacman -S ffmpeg
```

**on MacOS using Homebrew (https://brew.sh/)**
```bash
brew install ffmpeg
```

**on Windows using Chocolatey (https://chocolatey.org/)**
```bash
choco install ffmpeg
```

**on Windows using Scoop (https://scoop.sh/)**
```bash
scoop install ffmpeg
```

* If [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform, install [`rust`](http://rust-lang.org). Follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install the Rust development environment. 

### Installation

* You can download and install (or update to) the latest release of Whisper with the following command:

```bash
pip install -U openai-whisper
```

* Alternatively, use the following command to pull and install the latest commit from this repository and its Python dependencies:

```bash
pip install git+https://github.com/openai/whisper.git 
```

* To update the package to the latest version of this repository, run:

```bash
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
```

### Installation troubleshooting

If you see installation errors during the installation of Whisper, follow these steps:
* Check if you have [`rust`](http://rust-lang.org) installed on your system. If not, follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install the Rust development environment.
* Additionally, you may need to configure the `PATH` environment variable. You can use the following command: 
 ```bash
export PATH="$HOME/.cargo/bin:$PATH"
```
* If the installation fails with `No module named 'setuptools_rust'`, install `setuptools_rust`. You can use the following command:

```bash
pip install setuptools-rust
```

## Available models and languages

There are six model sizes, four with English-only versions, offering a compromise between speed and accuracy. In the table below are the names of the available models, their approximate memory requirements and their inference speed relative to the large model. The relative speeds given in the table are measured by transcribing English speech on the A100 graphics processing unit (GPU). The real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3`. It offers faster transcription speed with a minimal degradation in accuracy.

## Performance 

Whisper's performance varies widely by language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language. The performance breakdown uses Word Error Rates (WER) or Character Error Rates (CER, shown in *Italics*) evaluated on the Common Voice 15 and Fleurs datasets. Additional Word Error Rates or Character Error Rates metrics corresponding to the other models and datasets can be found in:
* Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356).
* The Bilingual Evaluation Understudy (BLEU) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## Command-line usage

The following command will transcribe speech in audio files. The command uses the `turbo` model:

    whisper audio.flac audio.mp3 audio.wav --model turbo

The default setting (which selects the `turbo` model) works well for transcribing English. To transcribe an audio file containing non-English speech, you can specify the language using the `--language` option:

    whisper japanese.wav --language Japanese

Add `--task translate` to translate the speech into English:

    whisper japanese.wav --language Japanese --task translate

Run the following command to view all available options:

    whisper --help

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.

## Python usage

Transcription can also be performed within Python: 

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window. The method performs autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model:

```python
import whisper

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

## More examples

Use the [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) category in Discussions for sharing more example usages of Whisper and third-party extensions such as web demos, integrations with other tools or ports for different platforms.

## License

Whisper's code and model weights are released under the Massachusetts Institute of Technology (MIT) License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
