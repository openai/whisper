import importlib

transcribe_mod = importlib.import_module("whisper.transcribe")
from whisper.transcribe import cli


def test_cli_threads_none_does_not_crash(monkeypatch, tmp_path):
    """`--threads None` parses to None via optional_int; cli() must not crash on it."""
    monkeypatch.setattr("whisper.load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        transcribe_mod,
        "transcribe",
        lambda model, audio, **kwargs: {"text": "", "segments": [], "language": "en"},
    )
    monkeypatch.setattr(
        transcribe_mod,
        "get_writer",
        lambda fmt, out_dir: (lambda result, audio_path, **kw: None),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "whisper",
            "dummy.wav",
            "--threads",
            "None",
            "--output_dir",
            str(tmp_path),
        ],
    )
    cli()  # must not raise TypeError: '>' not supported between 'NoneType' and 'int'


def test_cli_threads_zero_still_works(monkeypatch, tmp_path):
    """Default `--threads 0` behavior is unchanged (no set_num_threads call)."""
    monkeypatch.setattr("whisper.load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        transcribe_mod,
        "transcribe",
        lambda model, audio, **kwargs: {"text": "", "segments": [], "language": "en"},
    )
    monkeypatch.setattr(
        transcribe_mod,
        "get_writer",
        lambda fmt, out_dir: (lambda result, audio_path, **kw: None),
    )
    monkeypatch.setattr(
        "sys.argv", ["whisper", "dummy.wav", "--output_dir", str(tmp_path)]
    )
    cli()
