# CHANGELOG


## [v20230306](https://github.com/openai/whisper/releases/tag/v20230306)

* #1021: remove auxiliary audio extension
* #1038: apply formatting with `black`, `isort`, and `flake8`
* #869: word-level timestamps in `transcribe()`
* #1033: Decoding improvements
* #894: Update README.md
* #914: Fix infinite loop caused by incorrect timestamp tokens prediction
* #889: drop python 3.7 support

## [v20230124](https://github.com/openai/whisper/releases/tag/v20230124)

* #887: handle printing even if sys.stdout.buffer is not available
* #228: Add TSV formatted output in transcript, using integer start/end time in milliseconds
* #333: Added `--output_format` option
* #864: Handle `XDG_CACHE_HOME` properly for `download_root`
* #867: use stdout for printing transcription progress
* #659: Fix bug where mm is mistakenly replaced with hmm in e.g. 20mm
* #859: print '?' if a letter can't be encoded using the system default encoding

## [v20230117](https://github.com/openai/whisper/releases/tag/v20230117)

The first versioned release available on [PyPI](https://pypi.org/project/openai-whisper/)
