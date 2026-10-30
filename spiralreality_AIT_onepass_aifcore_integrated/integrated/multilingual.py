"""Utilities and curated corpora for multilingual boundary training."""

from __future__ import annotations

import random
import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

from .corpus import (
    TRAIN_TEXTS,
    register_teacher_segment,
    teacher_segments,
)
from .datasets import CorpusSample, languages as dataset_languages, samples_for_language
from .streaming import SegmentStream


@dataclass(frozen=True)
class LanguageCorpus:
    code: str
    texts: Tuple[str, ...]
    segments: Tuple[Tuple[str, ...], ...]
    samples: Tuple[CorpusSample, ...]


def _make_language_corpus(code: str) -> LanguageCorpus:
    samples = tuple(
        sample for sample in samples_for_language(code) if sample.group == "multilingual"
    )
    texts = tuple(sample.text for sample in samples)
    segs = tuple(sample.segments for sample in samples)
    return LanguageCorpus(code=code, texts=texts, segments=segs, samples=samples)


LANGUAGE_CORPORA: Dict[str, LanguageCorpus] = {}
for code in dataset_languages():
    corpus = _make_language_corpus(code)
    if corpus.texts:
        LANGUAGE_CORPORA[code] = corpus


AVAILABLE_LANGUAGES: Tuple[str, ...] = tuple(sorted(LANGUAGE_CORPORA.keys()))
# A first Slavic demo pack; these authored fixtures are separate from pinned
# external corpora and never serve as held-out learning-quality evidence.
SLAVIC_LANGUAGES: Tuple[str, ...] = ("uk", "pl", "cs")


def build_multilingual_corpus(
    languages: Sequence[str] | None = None,
    *,
    include_reflective: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Assemble training texts, segments, and language tags.

    Parameters
    ----------
    languages:
        Language codes to include. ``None`` selects all curated multilingual corpora.
    include_reflective:
        If ``True`` the default reflective English/Japanese corpus is included.
    shuffle:
        When ``True`` the combined dataset is shuffled deterministically using ``seed``.
    seed:
        Seed forwarded to :mod:`random` for dataset shuffling. ``None`` keeps the original
        ordering when ``shuffle`` is disabled.

    Returns
    -------
    texts, segments, tags:
        Lists of texts, segmentation labels, and per-text language tags. The tags match the
        ``languages`` argument with ``"reflective"`` reserved for the base corpus.
    """

    if languages is None:
        languages = AVAILABLE_LANGUAGES

    combined = list(_iter_multilingual_records(
        languages,
        include_reflective=include_reflective,
        shuffle=shuffle,
        seed=seed,
    ))

    texts = [text for _, text in combined]
    segments = teacher_segments(texts)
    tags = [lang for lang, _ in combined]
    return texts, [list(seg) for seg in segments], tags


def language_histogram(tags: Sequence[str]) -> Dict[str, int]:
    """Count occurrences of each language tag."""

    hist: Dict[str, int] = {}
    for tag in tags:
        hist[tag] = hist.get(tag, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: item[0]))


def language_statistics(
    texts: Sequence[str],
    segments: Sequence[Sequence[str]],
    tags: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute aggregate statistics for the multilingual dataset.

    Parameters
    ----------
    texts, segments, tags:
        Parallel sequences describing the dataset produced by
        :func:`build_multilingual_corpus`.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-language statistics including the sample ``count`` plus mean character
        lengths and segmentation granularity.
    """

    if not (len(texts) == len(segments) == len(tags)):
        raise ValueError("texts, segments, and tags must share the same length")

    aggregates: Dict[str, Dict[str, float]] = {}
    for text, seg, tag in zip(texts, segments, tags):
        seg_list = list(seg)
        char_count = float(len(text))
        token_count = float(len(seg_list))
        per_token = char_count if token_count == 0 else char_count / token_count
        if tag not in aggregates:
            aggregates[tag] = {
                "count": 0,
                "sum_chars": 0.0,
                "sum_tokens": 0.0,
                "sum_chars_per_token": 0.0,
            }
        agg = aggregates[tag]
        agg["count"] += 1
        agg["sum_chars"] += char_count
        agg["sum_tokens"] += token_count
        agg["sum_chars_per_token"] += per_token

    stats: Dict[str, Dict[str, float]] = {}
    for tag, agg in aggregates.items():
        count = int(agg["count"]) if agg["count"] else 0
        if count == 0:
            stats[tag] = {
                "count": 0.0,
                "mean_chars": 0.0,
                "mean_tokens": 0.0,
                "mean_chars_per_token": 0.0,
            }
            continue
        stats[tag] = {
            "count": float(count),
            "mean_chars": agg["sum_chars"] / count,
            "mean_tokens": agg["sum_tokens"] / count,
            "mean_chars_per_token": agg["sum_chars_per_token"] / count,
        }
    return dict(sorted(stats.items(), key=lambda item: item[0]))


def _iter_multilingual_records(
    languages: Sequence[str] | None,
    *,
    include_reflective: bool,
    shuffle: bool,
    seed: int | None,
) -> Iterator[Tuple[str, str]]:
    if languages is None:
        languages = AVAILABLE_LANGUAGES

    def base_records() -> Iterator[Tuple[str, str]]:
        if include_reflective:
            for text in TRAIN_TEXTS:
                yield "reflective", text
        for code in languages:
            corpus = LANGUAGE_CORPORA.get(code)
            if corpus is None:
                raise ValueError(f"Unknown language code: {code}")
            for text, seg in zip(corpus.texts, corpus.segments):
                register_teacher_segment(text, list(seg))
                yield code, text

    records = list(base_records()) if shuffle else base_records()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)  # type: ignore[arg-type]
        return iter(records)  # type: ignore[return-value]
    return records


def iter_multilingual_texts(
    languages: Sequence[str] | None = None,
    *,
    include_reflective: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[Tuple[str, str]]:
    """Yield ``(tag, text)`` pairs lazily for the multilingual corpus."""

    return _iter_multilingual_records(
        languages,
        include_reflective=include_reflective,
        shuffle=shuffle,
        seed=seed,
    )


def multilingual_segment_stream(
    languages: Sequence[str] | None = None,
    *,
    include_reflective: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
    chunk_size: int = 32,
    max_prefetch: int = 0,
    drop_incomplete: bool = False,
) -> SegmentStream:
    """Return a :class:`SegmentStream` covering the multilingual corpus.

    When ``shuffle`` is ``True`` the corpus is materialised before batching to
    ensure global ordering.  The default ``shuffle=False`` keeps the iterator
    fully streaming.
    ``drop_incomplete`` mirrors :class:`SegmentStream` by omitting the final
    partial batch when requested.
    """

    records = iter_multilingual_texts(
        languages,
        include_reflective=include_reflective,
        shuffle=shuffle,
        seed=seed,
    )
    text_iter, tag_iter = itertools.tee(records)
    texts = (text for _, text in text_iter)
    tags = (tag for tag, _ in tag_iter)
    return SegmentStream(
        texts,
        metadata=tags,
        chunk_size=chunk_size,
        max_prefetch=max_prefetch,
        drop_incomplete=drop_incomplete,
    )


__all__ = [
    "AVAILABLE_LANGUAGES",
    "SLAVIC_LANGUAGES",
    "LANGUAGE_CORPORA",
    "LanguageCorpus",
    "build_multilingual_corpus",
    "iter_multilingual_texts",
    "language_histogram",
    "language_statistics",
    "multilingual_segment_stream",
]

