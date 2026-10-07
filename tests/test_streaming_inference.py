from __future__ import annotations

import random
import re
from typing import Sequence

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming_inference import (
    ChunkedStreamingSegmenter,
)


def test_chunked_streaming_segmenter_respects_max_window() -> None:
    calls: list[int] = []

    def segmenter(text: str) -> Sequence[str]:
        calls.append(len(text))
        return [text]

    streamer = ChunkedStreamingSegmenter(
        segmenter,
        max_window_chars=16,
        lookahead_chars=4,
        context_chars=6,
        hard_split=True,
    )
    text = "abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    for chunk in ("abc", "defghijklmnop", "qrstuvwxyz"):
        out.extend(streamer.feed(chunk))
    out.extend(streamer.flush())
    assert "".join(out) == text
    assert calls and max(calls) <= 16


def test_onepassait_streaming_segmenter_roundtrip() -> None:
    ait = OnePassAIT(latent_dim=16, seed=123)
    streamer = ait.streaming_segmenter(
        max_window_chars=128,
        lookahead_chars=32,
        context_chars=64,
    )
    text = TRAIN_TEXTS[0]
    step = max(1, len(text) // 3)
    out: list[str] = []
    out.extend(streamer.feed(text[:step]))
    out.extend(streamer.feed(text[step : 2 * step]))
    out.extend(streamer.feed(text[2 * step :]))
    out.extend(streamer.flush())
    assert "".join(out) == text


def test_small_feeds_wait_for_real_lookahead() -> None:
    calls: list[str] = []

    def segmenter(text: str) -> Sequence[str]:
        calls.append(text)
        return [text]

    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=64, lookahead_chars=32,
    )
    for chunk in ("hel", "lo", " ", "world"):
        assert stream.feed(chunk) == []
    assert calls == []
    assert stream.pending_text == "hello world"
    assert stream.flush() == ["hello world"]
    assert stream.pending_text == ""


def test_unfinished_token_is_not_split_below_window_pressure() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: [text], max_window_chars=32, lookahead_chars=4,
    )
    assert stream.feed("abcdefghij") == []
    assert stream.feed("klmnopqrst") == []
    assert stream.flush() == ["abcdefghijklmnopqrst"]


@pytest.mark.parametrize("seed", range(10))
def test_word_tokens_survive_arbitrary_input_partitions(seed: int) -> None:
    def words(text: str) -> Sequence[str]:
        return re.findall(r"\w+|[^\w]", text)

    text = "Hello world! This stream keeps words intact. 日本語 🌀 e\u0301! " * 4
    stream = ChunkedStreamingSegmenter(
        words, max_window_chars=32, lookahead_chars=8, context_chars=16,
    )
    rng = random.Random(seed)
    emitted: list[str] = []
    cursor = 0
    while cursor < len(text):
        size = rng.randint(1, 41)
        emitted.extend(stream.feed(text[cursor:cursor + size]))
        cursor += size
    emitted.extend(stream.flush())
    assert emitted == list(words(text))


@pytest.mark.parametrize("hard_split", [False, True])
@pytest.mark.parametrize("window,lookahead,context", [(9, 0, 0), (9, 8, 100), (32, 8, 16)])
def test_random_partitions_preserve_exact_unicode_coverage(
    hard_split: bool, window: int, lookahead: int, context: int,
) -> None:
    text = "日本語 and whitespace\n\t🌀🐈\u200d⬛ e\u0301 repeated repeated " * 7
    for seed in range(8):
        calls: list[int] = []

        def characters(value: str) -> Sequence[str]:
            calls.append(len(value))
            return list(value)

        stream = ChunkedStreamingSegmenter(
            characters, max_window_chars=window, lookahead_chars=lookahead,
            context_chars=context, hard_split=hard_split,
        )
        rng = random.Random(seed)
        emitted: list[str] = []
        cursor = 0
        while cursor < len(text):
            size = rng.randint(1, 70)
            emitted.extend(stream.feed(text[cursor:cursor + size]))
            cursor += size
            assert len(stream.pending_text) <= window
            assert "".join(emitted) + stream.pending_text == text[:cursor]
        emitted.extend(stream.flush())
        assert "".join(emitted) == text
        assert max(calls) <= window


def test_huge_chunk_uses_bounded_pending_and_model_windows() -> None:
    observed: list[tuple[int, int]] = []

    def whole_token(text: str) -> Sequence[str]:
        observed.append((len(text), len(stream.pending_text)))
        return [text]

    stream = ChunkedStreamingSegmenter(
        whole_token, max_window_chars=32, lookahead_chars=8, context_chars=1000,
    )
    text = "abcdef🌀🐈" * 10000
    out = stream.feed(text)
    assert "".join(out) + stream.pending_text == text
    assert len(stream.pending_text) <= 32
    out.extend(stream.flush())
    assert "".join(out) == text
    assert all(model <= 32 and pending <= 32 for model, pending in observed)
    assert all(len(token) == 24 for token in out[:-1])


def test_left_context_yields_window_space_to_unfinished_token() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: re.findall(r"\S+\s*|\s+", text),
        max_window_chars=16, lookahead_chars=2, context_chars=16,
    )
    assert stream.feed("hello abc") == ["hello "]
    assert stream.feed("defghijkl") == []
    assert stream.flush() == ["abcdefghijkl"]


def test_hard_split_false_rejects_overflow_without_accepting_the_feed() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: [text], max_window_chars=16,
        lookahead_chars=4, hard_split=False,
    )
    assert stream.feed("abcdefghijklmnop") == []
    with pytest.raises(BufferError, match="flush"):
        stream.feed("qrst")
    assert stream.pending_text == "abcdefghijklmnop"
    assert stream.flush() == ["abcdefghijklmnop"]
    assert stream.feed("qrst") == []
    assert stream.flush() == ["qrst"]


def test_failed_huge_feed_rolls_back_previously_processed_windows() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: re.findall(r"\S+\s*|\s+", text),
        max_window_chars=16, lookahead_chars=4, hard_split=False,
    )
    stream.feed("hi ")
    with pytest.raises(BufferError):
        stream.feed("there " + "x" * 40)
    assert stream.pending_text == "hi "
    assert stream.flush() == ["hi "]


@pytest.mark.parametrize("bad_result", [[], ["wrong"], ["a", ""], [1], "abc", {}])
def test_invalid_segmenter_output_cannot_silently_lose_text(bad_result: object) -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: bad_result, max_window_chars=16, lookahead_chars=2,
    )
    stream.feed("ab")
    with pytest.raises(ValueError):
        stream.feed("c")
    assert stream.pending_text == "ab"
    with pytest.raises(ValueError):
        stream.flush()
    assert stream.pending_text == "ab"
    stream.segmenter = lambda text: [text]
    assert stream.flush() == ["ab"]


def test_failure_after_a_commit_rolls_back_the_entire_feed() -> None:
    calls = 0

    def failing(text: str) -> Sequence[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("backend failed")
        return list(text)

    stream = ChunkedStreamingSegmenter(
        failing, max_window_chars=16, lookahead_chars=4,
    )
    stream.feed("abc")
    with pytest.raises(RuntimeError, match="backend failed"):
        stream.feed("defghijklmnopqrstuvwxyz")
    assert stream.pending_text == "abc"
    stream.segmenter = list
    out = stream.feed("defghijklmnopqrstuvwxyz") + stream.flush()
    assert "".join(out) == "abcdefghijklmnopqrstuvwxyz"


def test_metadata_results_and_flush_without_reset() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: {"tokens": list(text)}, max_window_chars=16,
        lookahead_chars=4, context_chars=4,
    )
    assert stream.feed("abc") == []
    assert stream.flush(reset=False) == ["a", "b", "c"]
    assert stream.feed("def") == []
    assert stream.flush() == ["d", "e", "f"]
    assert stream.flush() == []


@pytest.mark.parametrize("lookahead", [16, 17, 100])
def test_lookahead_must_leave_room_for_progress(lookahead: int) -> None:
    with pytest.raises(ValueError, match="smaller"):
        ChunkedStreamingSegmenter(list, max_window_chars=16, lookahead_chars=lookahead)


def test_confidence_waits_then_releases_boundaries_with_more_context() -> None:
    def segmenter(text: str):
        probabilities = [0.0] * max(0, len(text) - 1)
        for index in range(1, len(text) - 1, 2):
            probabilities[index] = 0.8 if len(text) >= 6 else 0.4
        return {
            "tokens": [text[index:index + 2] for index in range(0, len(text), 2)],
            "boundary_probabilities": probabilities,
        }

    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=16, lookahead_chars=0,
        context_chars=4, min_boundary_confidence=0.8,
    )
    assert stream.feed("abcd") == []
    assert stream.pending_text == "abcd"
    assert stream.feed("ef") == ["ab", "cd"]
    assert stream.pending_text == "ef"
    # The boundary index is relative to the full model window, including context.
    assert stream.feed("gh") == ["ef"]
    assert stream.pending_text == "gh"
    assert stream.flush() == ["gh"]


def test_confidence_stops_at_first_uncertain_boundary() -> None:
    def segmenter(text: str):
        return {"tokens": list(text), "boundary_probabilities": [0.2] + [0.99] * (len(text) - 2)}

    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=16, lookahead_chars=0, min_boundary_confidence=0.8,
    )
    assert stream.feed("abcd") == []
    assert stream.pending_text == "abcd"
    # Final input commits valid tokens even when the marginal is uncertain.
    assert stream.flush() == list("abcd")


def test_confidence_waits_at_window_end_even_without_lookahead() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: {"tokens": [text], "boundary_probabilities": [1.0] * (len(text) - 1)},
        max_window_chars=16, lookahead_chars=0, min_boundary_confidence=0.1,
    )
    assert stream.feed("a") == []
    assert stream.feed("b") == []
    assert stream.flush() == ["ab"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), None, "0.5"])
def test_confidence_threshold_must_be_a_finite_probability(threshold) -> None:
    with pytest.raises(ValueError, match="min_boundary_confidence"):
        ChunkedStreamingSegmenter(list, min_boundary_confidence=threshold)


@pytest.mark.parametrize("probabilities", [
    None, 0.5, "00", {}, [], [0.5], [0.5, 0.5, 0.5],
    [float("nan"), 0.5], [float("inf"), 0.5], [-0.1, 0.5], [1.1, 0.5], ["0.9", 0.5],
])
def test_invalid_confidence_metadata_rolls_back_pending_text(probabilities) -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: {"tokens": list(text), "boundary_probabilities": probabilities},
        max_window_chars=16, lookahead_chars=2, min_boundary_confidence=0.8,
    )
    assert stream.feed("ab") == []
    with pytest.raises(ValueError, match="boundary_probabilities"):
        stream.feed("c")
    assert stream.pending_text == "ab"
    stream.segmenter = lambda text: {"tokens": list(text), "boundary_probabilities": [0.0] * (len(text) - 1)}
    assert stream.flush() == ["a", "b"]


@pytest.mark.parametrize("segmenter", [list, lambda text: {"tokens": list(text)}])
def test_confidence_requires_metadata_even_for_final_flush(segmenter) -> None:
    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=16, lookahead_chars=2, min_boundary_confidence=0.8,
    )
    stream.feed("ab")
    with pytest.raises(ValueError, match="boundary_probabilities"):
        stream.flush()
    assert stream.pending_text == "ab"


def test_confidence_failure_after_a_commit_rolls_back_entire_feed() -> None:
    calls = 0

    def segmenter(text: str):
        nonlocal calls
        calls += 1
        probabilities = [1.0] * (len(text) - 1)
        if calls == 2:
            probabilities[0] = float("nan")
        return {"tokens": list(text), "boundary_probabilities": probabilities}

    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=16, lookahead_chars=2,
        context_chars=4, min_boundary_confidence=0.8,
    )
    stream.feed("ab")
    with pytest.raises(ValueError, match="boundary_probabilities"):
        stream.feed("cdefghijklmnopqrstuvwxyz")
    assert stream.pending_text == "ab"
    assert stream._context == ""


@pytest.mark.parametrize("lookahead", [0, 4, 15])
def test_high_confidence_threshold_preserves_hard_window_bounds(lookahead: int) -> None:
    window_sizes: list[int] = []

    def segmenter(text: str):
        window_sizes.append(len(text))
        return {"tokens": list(text), "boundary_probabilities": [0.01] * (len(text) - 1)}

    stream = ChunkedStreamingSegmenter(
        segmenter, max_window_chars=16, lookahead_chars=lookahead,
        context_chars=64, min_boundary_confidence=1.0, hard_split=True,
    )
    text = "high confidence does not disable bounded safety 🌀" * 20
    emitted = stream.feed(text)
    assert "".join(emitted) + stream.pending_text == text
    assert len(stream.pending_text) <= 16
    emitted.extend(stream.flush())
    assert "".join(emitted) == text
    assert max(window_sizes) <= 16


def test_uncertain_boundaries_with_hard_split_disabled_reject_overflow() -> None:
    stream = ChunkedStreamingSegmenter(
        lambda text: {"tokens": list(text), "boundary_probabilities": [0.0] * (len(text) - 1)},
        max_window_chars=16, lookahead_chars=0,
        min_boundary_confidence=0.8, hard_split=False,
    )
    assert stream.feed("abcdefghijklmnop") == []
    with pytest.raises(BufferError):
        stream.feed("q")
    assert stream.pending_text == "abcdefghijklmnop"
    assert stream.flush() == list("abcdefghijklmnop")


def test_default_confidence_keeps_plain_list_segmenters_supported() -> None:
    stream = ChunkedStreamingSegmenter(
        list, max_window_chars=16, lookahead_chars=0,
    )
    assert stream.min_boundary_confidence == 0.0
    assert stream.feed("abc") == ["a", "b", "c"]
