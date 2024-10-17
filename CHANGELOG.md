# CHANGELOG

## [v20240930](https://github.com/openai/whisper/releases/tag/v20240930)

* allowing numpy 2 in tests ([#2362](https://github.com/openai/whisper/pull/2362))
* large-v3-turbo model ([#2361](https://github.com/openai/whisper/pull/2361))
* test on python/pytorch versions up to 3.12 and 2.4.1 ([#2360](https://github.com/openai/whisper/pull/2360))
* using sdpa if available ([#2359](https://github.com/openai/whisper/pull/2359))

## [v20240927](https://github.com/openai/whisper/releases/tag/v20240927)

* pinning numpy<2 in tests ([#2332](https://github.com/openai/whisper/pull/2332))
* Relax triton requirements for compatibility with pytorch 2.4 and newer ([#2307](https://github.com/openai/whisper/pull/2307))
* Skip silence around hallucinations ([#1838](https://github.com/openai/whisper/pull/1838))
* Fix triton env marker ([#1887](https://github.com/openai/whisper/pull/1887))

## [v20231117](https://github.com/openai/whisper/releases/tag/v20231117)

* Relax triton requirements for compatibility with pytorch 2.1 and newer ([#1802](https://github.com/openai/whisper/pull/1802))

## [v20231106](https://github.com/openai/whisper/releases/tag/v20231106)

* large-v3 ([#1761](https://github.com/openai/whisper/pull/1761))

## [v20231105](https://github.com/openai/whisper/releases/tag/v20231105)

* remove tiktoken pin ([#1759](https://github.com/openai/whisper/pull/1759))
* docs: Disambiguation of the term "relative speed" in the README ([#1751](https://github.com/openai/whisper/pull/1751))
* allow_pickle=False while loading of mel matrix IN audio.py ([#1511](https://github.com/openai/whisper/pull/1511))
* handling transcribe exceptions. ([#1682](https://github.com/openai/whisper/pull/1682))
* Add new option to generate subtitles by a specific number of words ([#1729](https://github.com/openai/whisper/pull/1729))
* Fix exception when an audio file with no speech is provided ([#1396](https://github.com/openai/whisper/pull/1396))

## [v20230918](https://github.com/openai/whisper/releases/tag/v20230918)

* Add .pre-commit-config.yaml ([#1528](https://github.com/openai/whisper/pull/1528))
* fix doc of TextDecoder ([#1526](https://github.com/openai/whisper/pull/1526))
* Update model-card.md ([#1643](https://github.com/openai/whisper/pull/1643))
* word timing tweaks ([#1559](https://github.com/openai/whisper/pull/1559))
* Avoid rearranging all caches ([#1483](https://github.com/openai/whisper/pull/1483))
* Improve timestamp heuristics. ([#1461](https://github.com/openai/whisper/pull/1461))
* fix condition_on_previous_text ([#1224](https://github.com/openai/whisper/pull/1224))
* Fix numba depreceation notice ([#1233](https://github.com/openai/whisper/pull/1233))
* Updated README.md to provide more insight on BLEU and specific appendices ([#1236](https://github.com/openai/whisper/pull/1236))
* Avoid computing higher temperatures on no_speech segments ([#1279](https://github.com/openai/whisper/pull/1279))
* Dropped unused execute bit from mel_filters.npz. ([#1254](https://github.com/openai/whisper/pull/1254))
* Drop ffmpeg-python dependency and call ffmpeg directly. ([#1242](https://github.com/openai/whisper/pull/1242))
* Python 3.11 ([#1171](https://github.com/openai/whisper/pull/1171))
* Update decoding.py ([#1219](https://github.com/openai/whisper/pull/1219))
* Update decoding.py ([#1155](https://github.com/openai/whisper/pull/1155))
* Update README.md to reference tiktoken ([#1105](https://github.com/openai/whisper/pull/1105))
* Implement max line width and max line count, and make word highlighting optional ([#1184](https://github.com/openai/whisper/pull/1184))
* Squash long words at window and sentence boundaries. ([#1114](https://github.com/openai/whisper/pull/1114))
* python-publish.yml: bump actions version to fix node warning ([#1211](https://github.com/openai/whisper/pull/1211))
* Update tokenizer.py ([#1163](https://github.com/openai/whisper/pull/1163))

## [v20230314](https://github.com/openai/whisper/releases/tag/v20230314)

* abort find_alignment on empty input ([#1090](https://github.com/openai/whisper/pull/1090))
* Fix truncated words list when the replacement character is decoded ([#1089](https://github.com/openai/whisper/pull/1089))
* fix github language stats getting dominated by jupyter notebook ([#1076](https://github.com/openai/whisper/pull/1076))
* Fix alignment between the segments and the list of words ([#1087](https://github.com/openai/whisper/pull/1087))
* Use tiktoken ([#1044](https://github.com/openai/whisper/pull/1044))

## [v20230308](https://github.com/openai/whisper/releases/tag/v20230308)

* kwargs in decode() for convenience ([#1061](https://github.com/openai/whisper/pull/1061))
* fix all_tokens handling that caused more repetitions and discrepancy in JSON ([#1060](https://github.com/openai/whisper/pull/1060))
* fix typo in CHANGELOG.md

## [v20230307](https://github.com/openai/whisper/releases/tag/v20230307)

* Fix the repetition/hallucination issue identified in #1046 ([#1052](https://github.com/openai/whisper/pull/1052))
* Use triton==2.0.0 ([#1053](https://github.com/openai/whisper/pull/1053))
* Install triton in x86_64 linux only ([#1051](https://github.com/openai/whisper/pull/1051))
* update setup.py to specify python >= 3.8 requirement

## [v20230306](https://github.com/openai/whisper/releases/tag/v20230306)

* remove auxiliary audio extension ([#1021](https://github.com/openai/whisper/pull/1021))
* apply formatting with `black`, `isort`, and `flake8` ([#1038](https://github.com/openai/whisper/pull/1038))
* word-level timestamps in `transcribe()` ([#869](https://github.com/openai/whisper/pull/869))
* Decoding improvements ([#1033](https://github.com/openai/whisper/pull/1033))
* Update README.md ([#894](https://github.com/openai/whisper/pull/894))
* Fix infinite loop caused by incorrect timestamp tokens prediction ([#914](https://github.com/openai/whisper/pull/914))
* drop python 3.7 support ([#889](https://github.com/openai/whisper/pull/889))

## [v20230124](https://github.com/openai/whisper/releases/tag/v20230124)

* handle printing even if sys.stdout.buffer is not available ([#887](https://github.com/openai/whisper/pull/887))
* Add TSV formatted output in transcript, using integer start/end time in milliseconds ([#228](https://github.com/openai/whisper/pull/228))
* Added `--output_format` option ([#333](https://github.com/openai/whisper/pull/333))
* Handle `XDG_CACHE_HOME` properly for `download_root` ([#864](https://github.com/openai/whisper/pull/864))
* use stdout for printing transcription progress ([#867](https://github.com/openai/whisper/pull/867))
* Fix bug where mm is mistakenly replaced with hmm in e.g. 20mm ([#659](https://github.com/openai/whisper/pull/659))
* print '?' if a letter can't be encoded using the system default encoding ([#859](https://github.com/openai/whisper/pull/859))

## [v20230117](https://github.com/openai/whisper/releases/tag/v20230117)

The first versioned release available on [PyPI](https://pypi.org/project/openai-whisper/)
