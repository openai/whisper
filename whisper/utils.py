import zlib
from typing import Iterator, TextIO


def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    return len(text) / len(zlib.compress(text.encode("utf-8")))


def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:" if hours > 0 else "") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def format_timestamp_for_srt(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    # hour is necessary for srt
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    """
    Write a transcript to a file in SRT format.

    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt

        result = transcribe(model, audio_path, temperature=temperature, **args)

        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for ind, segment in enumerate(transcript):
        # remove space at the start of the text
        if segment['text'][0] == ' ':
            segment['text'] = segment['text'][1:]

        # write srt lines
        print(f"{ind+1}", file=file)
        print(
            f"{format_timestamp_for_srt(segment['start'])} --> {format_timestamp_for_srt(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->')}\n",
            file=file,
            flush=True,
        )