import os

import pytest
import torch

import whisper


def test_progress_callback():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("tiny").to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "fdr.mp3")

    progress = []

    def callback(progress_data):
        progress.append(progress_data)

    model.transcribe(
        audio_path,
        language="en",
        verbose=False,  # purely for visualization purposes, not needed for the progress callback
        progress_callback=callback
    )
    print(progress)
    assert len(progress) > 0
    assert progress[-1] == 100.0
