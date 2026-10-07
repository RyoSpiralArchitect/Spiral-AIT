from __future__ import annotations

"""Optional external encoder adapters (Julia / R).

The NumPy transformer adapter remains the default, but this module probes for
externally implemented variants that expose a compatible Python API.  When
available they allow the rest of the system to leverage vendor-optimised
attention kernels while preserving the same end-to-end entry points.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class ExternalEncoderHandle:
    impl: Any
    backend: str = "julia"
    device: str = "cpu"
    available_devices: Tuple[str, ...] = ()

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - passthrough
        return self.impl.forward(*args, **kwargs)

    def tune_from_boundary(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        if hasattr(self.impl, "tune_from_boundary"):
            self.impl.tune_from_boundary(*args, **kwargs)

    def export_state(self) -> dict:  # pragma: no cover - passthrough
        if hasattr(self.impl, "export_state"):
            return dict(self.impl.export_state())
        if hasattr(self.impl, "state_dict"):
            return dict(self.impl.state_dict())
        return {}

    def load_state(self, state: dict) -> None:  # pragma: no cover - passthrough
        if hasattr(self.impl, "load_state"):
            self.impl.load_state(state)
        elif hasattr(self.impl, "load_state_dict"):
            self.impl.load_state_dict(state)

    def device_inventory(self) -> Tuple[str, ...]:
        """Return the device inventory advertised by the underlying backend."""

        if hasattr(self.impl, "device_inventory"):
            devices = self.impl.device_inventory()
            if isinstance(devices, Iterable):
                normalised = tuple(str(dev) for dev in devices)
                if normalised:
                    return normalised
        if self.available_devices:
            return self.available_devices
        return (self.device,)


@dataclass(frozen=True)
class BackendCandidate:
    module: str
    attr: str
    backend: str


_KNOWN_CANDIDATES: Tuple[BackendCandidate, ...] = (
    BackendCandidate("spiral_transformer_cpp", "CppTransformerAdapter", "cpp"),
    BackendCandidate("spiral_transformer_julia", "JuliaTransformerAdapter", "julia"),
    BackendCandidate("spiral_transformer_r", "RTransformerAdapter", "r"),
    BackendCandidate("spiralreality_transformer_cpp", "TransformerAdapter", "cpp"),
    BackendCandidate("spiralreality_transformer_julia", "TransformerAdapter", "julia"),
    BackendCandidate("spiralreality_transformer_r", "TransformerAdapter", "r"),
    BackendCandidate("spiral_transformer_adapter", "create_adapter", "generic"),
)


def _backend_preferences() -> Tuple[str, ...]:
    env_value = os.getenv("SPIRAL_ENCODER_BACKEND", "")
    if not env_value:
        return ()
    preferences = []
    for token in env_value.split(","):
        cleaned = token.strip().lower()
        if not cleaned or cleaned in {"auto", "any", "default"}:
            continue
        preferences.append(cleaned)
    return tuple(dict.fromkeys(preferences))  # Preserve order, drop duplicates.


def _candidate_modules() -> Tuple[BackendCandidate, ...]:
    preferences = _backend_preferences()
    if not preferences:
        return _KNOWN_CANDIDATES

    ordered: list[BackendCandidate] = []
    seen: set[BackendCandidate] = set()
    for pref in preferences:
        for candidate in _KNOWN_CANDIDATES:
            if candidate.backend == pref and candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
    for candidate in _KNOWN_CANDIDATES:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return tuple(ordered)


def _requested_device() -> Optional[str]:
    for key in (
        "SPIRAL_ENCODER_DEVICE",
        "SPIRAL_TRANSFORMER_DEVICE",
        "SPIRAL_DEVICE",
        "SPIRAL_DEFAULT_DEVICE",
    ):
        value = os.getenv(key)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _normalise_inventory(value: Any) -> Tuple[str, ...]:
    if isinstance(value, Iterable):
        result = tuple(str(item) for item in value)
        if result:
            return result
    return ()


def _apply_device_request(impl: Any, device: str) -> str:
    """Attempt to apply a device override to the instantiated adapter."""

    setters = (
        "set_device",
        "to_device",
        "to",
    )
    for name in setters:
        setter = getattr(impl, name, None)
        if callable(setter):
            try:
                setter(device)
                break
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Unable to set transformer device to {device!r}: {exc}"
                ) from exc
    else:
        if hasattr(impl, "device"):
            setattr(impl, "device", device)
        elif str(device).lower() not in {"cpu"}:
            raise ValueError("Device selection is not supported by this adapter")
    return str(getattr(impl, "device", device))


def load_external_adapter(d_model: int, n_layers: int, seed: int) -> Optional[ExternalEncoderHandle]:
    device_override = _requested_device()

    for candidate in _candidate_modules():
        try:
            module = importlib.import_module(candidate.module)
        except ModuleNotFoundError:
            continue
        factory = getattr(module, candidate.attr, None)
        if factory is None:
            LOGGER.warning(
                "Found module %s but missing %s attribute",
                candidate.module,
                candidate.attr,
            )
            continue
        kwargs = {"d_model": d_model, "n_layers": n_layers, "seed": seed}
        if device_override is not None:
            kwargs["device"] = device_override
        try:
            if callable(factory):
                try:
                    impl = factory(**kwargs)
                except TypeError:
                    if "device" in kwargs:
                        fallback_kwargs = dict(kwargs)
                        fallback_kwargs.pop("device", None)
                        impl = factory(**fallback_kwargs)
                    else:
                        raise
            else:
                impl = factory
        except TypeError:
            try:
                args = (d_model, n_layers, seed)
                if callable(factory):
                    if device_override is not None:
                        try:
                            impl = factory(*args, device=device_override)
                        except TypeError:
                            impl = factory(*args)
                    else:
                        impl = factory(*args)
                else:
                    impl = factory
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception(
                    "Failed to instantiate external encoder from %s", candidate.module
                )
                raise RuntimeError(f"Unable to instantiate external encoder: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception(
                "Failed to instantiate external encoder from %s", candidate.module
            )
            raise RuntimeError(f"Unable to instantiate external encoder: {exc}")

        if device_override is not None:
            device = _apply_device_request(impl, device_override)
        else:
            device = str(getattr(impl, "device", getattr(module, "DEFAULT_DEVICE", "cpu")))

        # Wrappers can delegate to NumPy when their compiled module is absent.
        # Report the implementation actually used, rather than the wrapper name.
        backend = getattr(impl, "backend", getattr(module, "BACKEND_KIND", candidate.backend))
        inventory = _normalise_inventory(getattr(module, "AVAILABLE_DEVICES", ()))
        if not inventory:
            inventory = _normalise_inventory(getattr(impl, "AVAILABLE_DEVICES", ()))
        if not inventory and hasattr(impl, "device_inventory"):
            inventory = _normalise_inventory(impl.device_inventory())
        if not inventory:
            inventory = (device,)

        LOGGER.info(
            "Using %s transformer adapter from %s on %s",
            backend,
            candidate.module,
            device,
        )
        return ExternalEncoderHandle(
            impl=impl,
            backend=str(backend),
            device=str(device),
            available_devices=inventory,
        )
    return None


def has_external_adapter() -> bool:
    for candidate in _candidate_modules():
        try:
            importlib.import_module(candidate.module)
            return True
        except ModuleNotFoundError:
            continue
    return False
