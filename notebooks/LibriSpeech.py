#!/usr/bin/env python
# coding: utf-8

# # Installing Whisper
# 
# The commands below will install the Python packages needed to use Whisper models and evaluate the transcription results.

# In[1]:


get_ipython().system(' pip install git+https://github.com/openai/whisper.git')
get_ipython().system(' pip install jiwer')


# # Loading the LibriSpeech dataset
# 
# The following will load the test-clean split of the LibriSpeech corpus using torchaudio.

# In[2]:


import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import whisper
import torchaudio

from tqdm.notebook import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)


# In[4]:


dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)


# # Running inference on the dataset using a base Whisper model
# 
# The following will take a few minutes to transcribe all utterances in the dataset.

# In[5]:


model = whisper.load_model("base.en")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


# In[6]:


# predict without timestamps for short-form transcription
options = whisper.DecodingOptions(language="en", without_timestamps=True)


# In[7]:


hypotheses = []
references = []

for mels, texts in tqdm(loader):
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)


# In[8]:


data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
data


# # Calculating the word error rate
# 
# Now, we use our English normalizer implementation to standardize the transcription and calculate the WER.

# In[9]:


import jiwer
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


# In[10]:


data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
data


# In[11]:


wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")

