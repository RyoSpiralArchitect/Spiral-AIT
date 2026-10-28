import builtins

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.utils import (
    seeded_vector,
)


def test_seeded_vector_is_deterministic() -> None:
    seeded_vector.cache_clear()
    reference = seeded_vector("unit-test", dim=8)
    seeded_vector.cache_clear()
    repeat = seeded_vector("unit-test", dim=8)
    np.testing.assert_allclose(reference, repeat)


def test_seeded_vector_does_not_depend_on_python_hash(monkeypatch) -> None:
    seeded_vector.cache_clear()

    def _failing_hash(_: object) -> int:
        raise AssertionError("unexpected hash call")

    # Restore the process-wide builtin before pytest hashes its own reports.
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "hash", _failing_hash)
        vector = seeded_vector("unit-test", dim=4)
    assert vector.shape == (4,)


def test_seeded_vector_supports_binary_names() -> None:
    seeded_vector.cache_clear()
    as_text = seeded_vector("unit-test", dim=6)
    as_bytes = seeded_vector(b"unit-test", dim=6)
    np.testing.assert_allclose(as_text, as_bytes)


def test_seeded_vector_dtype_and_immutability() -> None:
    seeded_vector.cache_clear()
    vector = seeded_vector("unit-test", dim=3, dtype=np.float16)
    assert vector.dtype == np.float16
    with pytest.raises(ValueError):
        vector[0] = 0.0
