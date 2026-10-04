from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """Replace a checkpoint only after its complete JSON payload is written."""

    serialised = json.dumps(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as fh:
            temporary = Path(fh.name)
            fh.write(serialised)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
