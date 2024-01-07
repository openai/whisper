This directory supplements the paper with more details on how we prepared the data for evaluation, to help replicate our experiments. 

## Short-form English-only datasets

### LibriSpeech

We used the test-clean and test-other splits from the [LibriSpeech ASR corpus](https://www.openslr.org/12).

### TED-LIUM 3

We used the test split of [TED-LIUM Release 3](https://www.openslr.org/51/), using the segmented manual transcripts included in the release.

### Common Voice 5.1

We downloaded the English subset of Common Voice Corpus 5.1 from [the official website](https://commonvoice.mozilla.org/en/datasets)

### Artie

We used the [Artie bias corpus](https://github.com/artie-inc/artie-bias-corpus). This is a subset of the Common Voice dataset.

### CallHome & Switchboard

We used the two corpora from [LDC2002S09](https://catalog.ldc.upenn.edu/LDC2002S09) and [LDC2002T43](https://catalog.ldc.upenn.edu/LDC2002T43) and followed the [eval2000_data_prep.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/fisher_swbd/s5/local/eval2000_data_prep.sh) script for preprocessing. The `wav.scp` files can be converted to WAV files with the following bash commands:

```bash
mkdir -p wav
while read name cmd; do
    echo $name
    echo ${cmd/\|/} wav/$name.wav | bash
done < wav.scp
```


### WSJ

We used [LDC93S6B](https://catalog.ldc.upenn.edu/LDC93S6B) and [LDC94S13B](https://catalog.ldc.upenn.edu/LDC94S13B) and followed the [s5 recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj/s5) to preprocess the dataset.

### CORAAL

We used the 231 interviews from [CORAAL (v. 2021.07)](https://oraal.uoregon.edu/coraal) and used the segmentations from [the FairSpeech project](https://github.com/stanford-policylab/asr-disparities/blob/master/input/CORAAL_transcripts.csv).

### CHiME-6

We downloaded the [CHiME-5 dataset](https://spandh.dcs.shef.ac.uk//chime_challenge/CHiME5/download.html) and followed the stage 0 of the [s5_track1 recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1) to create the CHiME-6 dataset which fixes synchronization. We then used the binaural recordings (`*_P??.wav`) and the corresponding transcripts.

### AMI-IHM, AMI-SDM1

We preprocessed the [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) by following the stage 0 ad 2 of the [s5b recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/ami/s5b).


## Long-form English-only datasets

### TED-LIUM 3

To create a long-form transcription dataset from the [TED-LIUM3](https://www.openslr.org/51/) dataset, we sliced the audio between the beginning of the first labeled segment and the end of the last labeled segment of each talk, and we used the concatenated text as the label. Below are the timestamps used for slicing each of the 11 TED talks in the test split.   

| Filename            | Begin time (s) | End time (s) |
|---------------------|----------------|--------------|
| DanBarber_2010      | 16.09          | 1116.24      |
| JaneMcGonigal_2010  | 15.476         | 1187.61      |
| BillGates_2010      | 15.861         | 1656.94      |
| TomWujec_2010U      | 16.26          | 402.17       |
| GaryFlake_2010      | 16.06          | 367.14       |
| EricMead_2009P      | 18.434         | 536.44       |
| MichaelSpecter_2010 | 16.11          | 979.312      |
| DanielKahneman_2010 | 15.8           | 1199.44      |
| AimeeMullins_2009P  | 17.82          | 1296.59      |
| JamesCameron_2010   | 16.75          | 1010.65      |
| RobertGupta_2010U   | 16.8           | 387.03       |

### Meanwhile

This dataset consists of 64 segments from The Late Show with Stephen Colbert. The YouTube video ID, start and end timestamps, and the labels can be found in [meanwhile.json](meanwhile.json). The labels are collected from the closed-caption data for each video and corrected with manual inspection.

### Rev16

We use a subset of 16 files from the 30 podcast episodes in [Rev.AI's Podcast Transcription Benchmark](https://www.rev.ai/blog/podcast-transcription-benchmark-part-1/), after finding that there are multiple cases where a significant portion of the audio and the labels did not match, mostly on the parts introducing the sponsors. We selected 16 episodes that do not have this error, whose "file number" are:

    3 4 9 10 11 14 17 18 20 21 23 24 26 27 29 32

### Kincaid46

This dataset consists of 46 audio files and the corresponding transcripts compiled in the blog article [Which automatic transcription service is the most accurate - 2018](https://medium.com/descript/which-automatic-transcription-service-is-the-most-accurate-2018-2e859b23ed19) by Jason Kincaid. We used the 46 audio files and reference transcripts from the Airtable widget in the article.

For the human transcription benchmark in the paper, we use a subset of 25 examples from this data, whose "Ref ID" are:

    2 4 5 8 9 10 12 13 14 16 19 21 23 25 26 28 29 30 33 35 36 37 42 43 45

### Earnings-21, Earnings-22

For these datasets, we used the files available in [the speech-datasets repository](https://github.com/revdotcom/speech-datasets), as of their `202206` version.

### CORAAL

We used the 231 interviews from [CORAAL (v. 2021.07)](https://oraal.uoregon.edu/coraal) and used the full-length interview files and transcripts.


## Multilingual datasets

### Multilingual LibriSpeech

We used the test splits from each language in [the Multilingual LibriSpeech (MLS) corpus](https://www.openslr.org/94/).

### Fleurs

We collected audio files and transcripts using the implementation available as [HuggingFace datasets](https://huggingface.co/datasets/google/fleurs/blob/main/fleurs.py). To use as a translation dataset, we matched the numerical utterance IDs to find the corresponding transcript in English.   

### VoxPopuli

We used the `get_asr_data.py` script from [the official repository](https://github.com/facebookresearch/voxpopuli) to collect the ASR data in 14 languages. 

### Common Voice 9

We downloaded the Common Voice Corpus 9 from [the official website](https://commonvoice.mozilla.org/en/datasets)

### CoVOST 2

We collected the `X into English` data collected using [the official repository](https://github.com/facebookresearch/covost).
