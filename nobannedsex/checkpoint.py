"""Versioned, atomically replaced checkpoints containing tensors and primitives."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch

FORMAT_VERSION = 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: str | Path) -> dict:
    # Restricted loader; no arbitrary object unpickling or allowlisted classes.
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cp, dict) or "model_config" not in cp or "model" not in cp:
        raise ValueError("not a NoBannedSex checkpoint")
    if cp.get("format_version", 1) not in (1, FORMAT_VERSION):
        raise ValueError("unsupported checkpoint format version")
    return cp


def atomic_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
