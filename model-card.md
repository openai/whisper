# Model Card: Whisper

This is the official codebase for running the automatic speech recognition (ASR) models (Whisper models) trained and released by OpenAI.

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993), we're providing some information about the automatic speech recognition model. More information on how these models were trained and evaluated can be found [in the paper](https://arxiv.org/abs/2212.04356).


## Model Details

The Whisper models are trained for speech recognition and translation tasks, capable of transcribing speech audio into the text in the language it is spoken (ASR) as well as translated into English (speech translation). Researchers at OpenAI developed the models to study the robustness of speech processing systems trained under large-scale weak supervision. There are 9 models of different sizes and capabilities, summarized in the following table.

|  Size  | Parameters | English-only model | Multilingual model |  
|:------:|:----------:|:------------------:|:------------------:|
|  tiny  |    39 M    |         ✓          |         ✓          |
|  base  |    74 M    |         ✓          |         ✓          |
| small  |   244 M    |         ✓          |         ✓          |
| medium |   769 M    |         ✓          |         ✓          |
| large  |   1550 M   |                    |         ✓          |

In December 2022, we [released an improved large model named `large-v2`](https://github.com/openai/whisper/discussions/661), and `large-v3` in November 2023.


### Release date

September 2022 (original series), December 2022 (`large-v2`), and November 2023 (`large-v3`)

### Model type

Sequence-to-sequence ASR (automatic speech recognition) and speech translation model

### Paper & samples

[Paper](https://arxiv.org/abs/2212.04356) / [Blog](https://openai.com/blog/whisper)


## Model Use

### Evaluated Use

The primary intended users of these models are AI researchers studying the robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.

The models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.

In particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes.


## Training Data

The models are trained on 680,000 hours of audio and the corresponding transcripts collected from the internet. 65% of this data (or 438,000 hours) represents English-language audio and matched English transcripts, roughly 18% (or 126,000 hours) represents non-English audio and English transcripts, while the final 17% (or 117,000 hours) represents non-English audio and the corresponding transcript. This non-English data represents 98 different languages. 

As discussed in [the accompanying paper](https://arxiv.org/abs/2212.04356), we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.


## Performance and Limitations

Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, and technical language, as well as zero-shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. 

However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.

Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include a higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in [the paper accompanying this release](https://arxiv.org/abs/2212.04356).

In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis of these limitations is provided in [the paper](https://arxiv.org/abs/2212.04356). It is likely that this behavior and hallucinations may be worse in lower-resource and/or lower-discoverability languages.


## Broader Implications

We anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications.

There are also potential dual-use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects.
