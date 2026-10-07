from __future__ import annotations

import sys
import types

from spiralreality_AIT_onepass_aifcore_integrated.integrated import encoder_backends


def test_wrapper_metadata_reports_actual_fallback(monkeypatch):
    import types
    import sys

    module_name = "test_numpy_fallback_adapter"
    module = types.ModuleType(module_name)
    module.BACKEND_KIND = "cpp"
    module.DEFAULT_DEVICE = "cuda"
    module.create_adapter = lambda **kwargs: types.SimpleNamespace(backend="spectral-numpy", device="cpu")
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(encoder_backends, "_KNOWN_CANDIDATES", (
        encoder_backends.BackendCandidate(module_name, "create_adapter", "cpp"),
    ))
    for name in ("SPIRAL_ENCODER_DEVICE", "SPIRAL_TRANSFORMER_DEVICE", "SPIRAL_DEVICE", "SPIRAL_DEFAULT_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    handle = encoder_backends.load_external_adapter(16, 1, 0)
    assert handle.backend == "spectral-numpy"
    assert handle.device == "cpu"


def _make_candidate(module_name: str, attr: str, backend: str) -> encoder_backends.BackendCandidate:
    return encoder_backends.BackendCandidate(module_name, attr, backend)


def test_backend_preferences_respect_env(monkeypatch):
    candidates = (
        _make_candidate("module_alpha", "Adapter", "alpha"),
        _make_candidate("module_beta", "Adapter", "beta"),
        _make_candidate("module_gamma", "Adapter", "gamma"),
    )
    monkeypatch.setattr(encoder_backends, "_KNOWN_CANDIDATES", candidates)
    monkeypatch.setenv("SPIRAL_ENCODER_BACKEND", "gamma, alpha, gamma, beta")

    ordered = encoder_backends._candidate_modules()

    assert [candidate.backend for candidate in ordered] == [
        "gamma",
        "alpha",
        "beta",
    ]


def test_load_external_adapter_device_override_keyword(monkeypatch):
    module_name = "spiral_transformer_adapter"
    module = types.ModuleType(module_name)
    calls: dict[str, str] = {}

    class Adapter:
        def __init__(self, *, d_model: int, n_layers: int, seed: int, device: str = "cpu"):
            self.backend = "kw"
            self.device = device
            self._devices = ("cpu", "cuda")
            calls["device"] = device

        def device_inventory(self):
            return self._devices

    module.create_adapter = Adapter
    module.BACKEND_KIND = "kw"
    module.DEFAULT_DEVICE = "cpu"
    module.AVAILABLE_DEVICES = ("cpu", "cuda")

    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        encoder_backends,
        "_KNOWN_CANDIDATES",
        (_make_candidate(module_name, "create_adapter", "kw"),),
    )
    monkeypatch.setenv("SPIRAL_ENCODER_DEVICE", "cuda")

    handle = encoder_backends.load_external_adapter(32, 2, 0)

    assert handle.device == "cuda"
    assert handle.device_inventory() == ("cpu", "cuda")
    assert calls["device"] == "cuda"


def test_load_external_adapter_device_override_without_keyword(monkeypatch):
    module_name = "spiral_transformer_cpp"
    module = types.ModuleType(module_name)

    class Adapter:
        def __init__(self, d_model: int, n_layers: int, seed: int):
            self.backend = "cpp"
            self.device = "cpu"
            self._devices = ("cpu", "cuda")

        def set_device(self, device: str) -> None:
            self.device = device

        def device_inventory(self):
            return self._devices

    module.CppTransformerAdapter = Adapter
    module.BACKEND_KIND = "cpp"
    module.DEFAULT_DEVICE = "cpu"

    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        encoder_backends,
        "_KNOWN_CANDIDATES",
        (_make_candidate(module_name, "CppTransformerAdapter", "cpp"),),
    )
    monkeypatch.setenv("SPIRAL_ENCODER_DEVICE", "cuda")

    handle = encoder_backends.load_external_adapter(32, 2, 0)

    assert handle.device == "cuda"
    assert handle.device_inventory() == ("cpu", "cuda")
