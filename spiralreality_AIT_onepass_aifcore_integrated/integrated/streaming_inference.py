"""Chunked streaming inference helpers for text segmentation.

The main model APIs operate on full strings which can become expensive for long
streams (especially when contextual encoders scale quadratically in sequence
length). This module provides a small stateful wrapper that:

- Maintains a rolling context suffix for feature quality near chunk boundaries.
- Limits the maximum window length passed to the underlying segmenter.
- Commits tokens incrementally while keeping a configurable lookahead tail.

The implementation is intentionally lightweight and model-agnostic: callers
provide a ``segmenter(text) -> tokens`` callable (for example
``OnePassAIT.segment_text``).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Dict, List, Sequence, Union

SegmenterFn = Callable[[str], Union[Sequence[str], Dict[str, object]]]


class ChunkedStreamingSegmenter:
    """Stateful text segmentation with bounded context and pending buffers.

    ``lookahead_chars`` are retained until more input arrives or :meth:`flush`
    marks the end of the input. Tokens crossing that tail are retained whole.
    Left context is shortened when necessary to give a long pending token the
    full ``max_window_chars`` budget.

    With ``hard_split=True``, an otherwise uncommittable token is split only
    when the pending buffer fills the entire window. With ``hard_split=False``
    it stays intact; accepting further input would raise ``BufferError`` and
    the caller must flush or reset before retrying. Failed feeds and flushes
    leave this wrapper's buffers unchanged. The segmenter must return nonempty
    string tokens that reconstruct its input exactly.

    ``min_boundary_confidence > 0`` additionally waits for the first uncertain
    boundary before committing any later tokens. The segmenter must then return
    a dictionary containing ``tokens`` and ``boundary_probabilities``: one finite
    probability in [0, 1] after each character except the last. End-of-window
    tokens wait for more input because they have no following boundary marginal.
    :meth:`flush` ignores the threshold, but still validates the metadata.
    Confidence does not override the bounded-buffer policy: ``hard_split=True``
    may force a split under full-window pressure even below the threshold.
    """

    def __init__(
        self,
        segmenter: SegmenterFn,
        *,
        max_window_chars: int = 512,
        lookahead_chars: int = 64,
        context_chars: int = 128,
        hard_split: bool = True,
        min_boundary_confidence: float = 0.0,
    ) -> None:
        if max_window_chars <= 8:
            raise ValueError("max_window_chars must be > 8")
        if lookahead_chars < 0:
            raise ValueError("lookahead_chars must be >= 0")
        if lookahead_chars >= max_window_chars:
            raise ValueError("lookahead_chars must be smaller than max_window_chars")
        if context_chars < 0:
            raise ValueError("context_chars must be >= 0")
        if (
            not isinstance(min_boundary_confidence, Real)
            or not math.isfinite(min_boundary_confidence)
            or not 0.0 <= min_boundary_confidence <= 1.0
        ):
            raise ValueError("min_boundary_confidence must be finite and between zero and one")
        self.segmenter = segmenter
        self.max_window_chars = int(max_window_chars)
        self.lookahead_chars = int(lookahead_chars)
        self.context_chars = int(context_chars)
        self.hard_split = bool(hard_split)
        self.min_boundary_confidence = float(min_boundary_confidence)
        self._context: str = ""
        self._pending: str = ""
        self._forced_split_count = 0

    @property
    def pending_text(self) -> str:
        return self._pending

    @property
    def forced_split_count(self) -> int:
        """Committed capacity-forced cuts since reset; preserved across flush."""
        return self._forced_split_count

    def reset(self) -> None:
        self._context = ""
        self._pending = ""
        self._forced_split_count = 0

    def feed(self, chunk: str) -> List[str]:
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if not chunk:
            return []
        initial = self._context, self._pending, self._forced_split_count
        out: List[str] = []
        offset = 0
        try:
            while offset < len(chunk):
                capacity = self.max_window_chars - len(self._pending)
                if capacity == 0:
                    raise BufferError(
                        "No token boundary fits the streaming window; flush or "
                        "reset before retrying, or enable hard_split"
                    )
                end = min(len(chunk), offset + capacity)
                self._pending += chunk[offset:end]
                offset = end
                out.extend(self._drain(final=False))
        except BaseException:
            self._context, self._pending, self._forced_split_count = initial
            raise
        return out

    def flush(self, *, reset: bool = True) -> List[str]:
        initial = self._context, self._pending, self._forced_split_count
        try:
            out = self._drain(final=True)
        except BaseException:
            self._context, self._pending, self._forced_split_count = initial
            raise
        if reset:
            self._context = self._pending = ""
        return out

    def _context_text(self) -> str:
        available = self.max_window_chars - len(self._pending)
        return self._context[-available:] if available > 0 else ""

    def _drain(self, *, final: bool) -> List[str]:
        out: List[str] = []
        while self._pending:
            committed = self._commit_once(final=final)
            if not committed:
                break
            out.extend(committed)
        return out

    def _commit_once(self, *, final: bool) -> List[str]:
        if not self._pending:
            return []
        if not final and len(self._pending) <= self.lookahead_chars:
            return []
        context_text = self._context_text()
        context_len = len(context_text)
        work_text = context_text + self._pending

        result = self.segmenter(work_text)
        boundary_probabilities: List[float] = []
        if self.min_boundary_confidence > 0.0:
            if not isinstance(result, dict) or "boundary_probabilities" not in result:
                raise ValueError("confidence-aware segmenter must return tokens and boundary_probabilities")
            raw_probabilities = result["boundary_probabilities"]
            if isinstance(raw_probabilities, (str, bytes, dict)):
                raise ValueError("boundary_probabilities must be a sequence of finite probabilities")
            try:
                values = list(raw_probabilities)
            except TypeError as exc:
                raise ValueError("boundary_probabilities must be a sequence of finite probabilities") from exc
            if len(values) != max(0, len(work_text) - 1):
                raise ValueError("boundary_probabilities must have one entry per internal character boundary")
            if any(
                not isinstance(value, Real) or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in values
            ):
                raise ValueError("boundary_probabilities must be finite and between zero and one")
            boundary_probabilities = [float(value) for value in values]
        if isinstance(result, dict):
            if "tokens" not in result:
                raise ValueError("segmenter result must contain tokens")
            result = result["tokens"]
        if isinstance(result, (str, bytes)):
            raise ValueError("segmenter must return a sequence of string tokens")
        tokens = list(result)
        if any(not isinstance(tok, str) or not tok for tok in tokens):
            raise ValueError("segmenter tokens must be nonempty strings")
        if "".join(tokens) != work_text:
            raise ValueError("segmenter tokens must reconstruct the input exactly")

        effective_lookahead = 0 if final else self.lookahead_chars
        commit_limit = len(work_text) - effective_lookahead
        commit_limit = max(context_len, commit_limit)

        committed: List[str] = []
        pos = 0
        for tok in tokens:
            start = pos
            end = pos + len(tok)
            pos = end
            if end <= context_len:
                continue
            if start < context_len < end:
                tok = tok[context_len - start :]
                start = context_len
            if end <= commit_limit:
                if not final and self.min_boundary_confidence > 0.0:
                    if end == len(work_text) or boundary_probabilities[end - 1] < self.min_boundary_confidence:
                        break
                if tok:
                    committed.append(tok)
            else:
                break

        if (
            not committed
            and self.hard_split
            and len(self._pending) == self.max_window_chars
            and commit_limit > context_len
        ):
            forced = work_text[context_len:commit_limit]
            if forced:
                committed = [forced]
                self._forced_split_count += 1

        if not committed:
            return []

        committed_chars = sum(len(tok) for tok in committed)
        self._pending = self._pending[committed_chars:]
        keep = min(self.context_chars, self.max_window_chars)
        self._context = (self._context + "".join(committed))[-keep:] if keep else ""
        return committed


__all__ = ["ChunkedStreamingSegmenter"]
