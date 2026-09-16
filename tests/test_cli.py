import os
import sys

import pytest

from whisper.transcribe import _missing_audio_files, cli


def test_missing_audio_files(tmp_path):
    existing = tmp_path / "clip.wav"
    existing.write_bytes(b"")
    missing = tmp_path / "missing.wav"

    assert _missing_audio_files([str(existing)]) == []
    assert _missing_audio_files([str(missing)]) == [str(missing)]
    assert _missing_audio_files([str(tmp_path)]) == [str(tmp_path)]
    assert _missing_audio_files([str(existing), str(missing)]) == [str(missing)]


def test_cli_rejects_missing_audio_before_loading_model(monkeypatch, capsys):
    missing = os.path.join("missing-dir", "missing.m4a")
    monkeypatch.setattr(sys, "argv", ["whisper", missing])

    called = []

    def fake_load_model(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("load_model should not run for a missing audio path")

    monkeypatch.setattr("whisper.load_model", fake_load_model)

    with pytest.raises(SystemExit) as excinfo:
        cli()

    assert excinfo.value.code == 2
    assert called == []
    assert missing in capsys.readouterr().err
