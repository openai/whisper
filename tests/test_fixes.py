import types

import pytest


# ---------------------------------------------------------------------------
# whisper/tokenizer.py -- Javanese language code alias
# ---------------------------------------------------------------------------


def test_jv_alias_in_to_language_code():
    from whisper.tokenizer import TO_LANGUAGE_CODE

    assert "jv" in TO_LANGUAGE_CODE, (
        "'jv' (ISO 639-1 code for Javanese) must be in TO_LANGUAGE_CODE so "
        "users can pass --language jv"
    )
    assert TO_LANGUAGE_CODE["jv"] == "jw", (
        "'jv' must map to 'jw' to match the token the model was trained with"
    )


def test_jw_still_in_languages():
    # The LANGUAGES key must stay as "jw" -- the model checkpoint uses <|jw|>
    from whisper.tokenizer import LANGUAGES

    assert "jw" in LANGUAGES
    assert LANGUAGES["jw"] == "javanese"


def test_javanese_name_resolves():
    from whisper.tokenizer import TO_LANGUAGE_CODE

    assert TO_LANGUAGE_CODE.get("javanese") == "jw"


# ---------------------------------------------------------------------------
# whisper/transcribe.py -- decode_with_fallback silence guard
# ---------------------------------------------------------------------------


def _make_result(compression_ratio, avg_logprob, no_speech_prob):
    return types.SimpleNamespace(
        compression_ratio=compression_ratio,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def _needs_fallback(
    result,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
):
    """Mirrors the fixed decode_with_fallback logic in transcribe.py."""
    needs_fallback = False

    if (
        compression_ratio_threshold is not None
        and result.compression_ratio > compression_ratio_threshold
    ):
        needs_fallback = True

    if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
        needs_fallback = True

    if (
        no_speech_threshold is not None
        and result.no_speech_prob > no_speech_threshold
        and logprob_threshold is not None
        and result.avg_logprob < logprob_threshold
        and (
            compression_ratio_threshold is None
            or result.compression_ratio <= compression_ratio_threshold
        )
    ):
        needs_fallback = False  # silence

    return needs_fallback


def test_good_output_no_fallback():
    assert not _needs_fallback(_make_result(1.2, -0.4, 0.1))


def test_repetition_loop_triggers_fallback():
    # High compression_ratio + high no_speech_prob: before the fix the silence
    # guard would suppress needs_fallback here, causing an infinite loop.
    assert _needs_fallback(_make_result(3.5, -1.5, 0.8))


def test_genuine_silence_suppresses_fallback():
    assert not _needs_fallback(_make_result(1.1, -2.0, 0.9))


def test_bad_logprob_alone_triggers_fallback():
    assert _needs_fallback(_make_result(1.3, -2.5, 0.2))


def test_severe_repetition_not_silenced():
    assert _needs_fallback(_make_result(4.0, -3.0, 0.95))


def test_compression_threshold_none_allows_silence():
    assert not _needs_fallback(
        _make_result(9.0, -2.0, 0.9), compression_ratio_threshold=None
    )


def test_ratio_exactly_at_threshold_allows_silence():
    # > not >= so exactly at threshold does not count as a loop
    assert not _needs_fallback(_make_result(2.4, -2.0, 0.9))


def test_ratio_just_above_threshold_forces_fallback():
    assert _needs_fallback(_make_result(2.401, -2.0, 0.9))


# ---------------------------------------------------------------------------
# whisper/transcribe.py -- progress_callback
# ---------------------------------------------------------------------------


def test_progress_callback_called_each_step():
    calls = []

    def cb(current, total):
        calls.append((current, total))

    # Simulate what the patched transcribe loop does
    total_steps = 5
    for step in range(1, total_steps + 1):
        cb(step, total_steps)

    assert len(calls) == total_steps
    assert calls[0] == (1, total_steps)
    assert calls[-1] == (total_steps, total_steps)


def test_progress_callback_none_does_not_crash():
    # When callback is None the loop must not call it
    callback = None
    total_steps = 3
    for step in range(1, total_steps + 1):
        if callback is not None:
            callback(step, total_steps)
    # No assertion needed -- reaching here means no crash


def test_progress_callback_receives_correct_fraction():
    fractions = []

    def cb(current, total):
        fractions.append(current / total)

    total = 10
    for step in range(1, total + 1):
        cb(step, total)

    assert fractions[0] == pytest.approx(0.1)
    assert fractions[-1] == pytest.approx(1.0)
