"""Cerebrium train-module entrypoint with runtime dependency bootstrapping."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

RUNTIME_PACKAGES = [
    "numpy>=1.26,<3",
    "pandas>=2.1,<3",
    "torch>=2.2,<3",
]


def _install_runtime_packages() -> None:
    """Install training dependencies inside a bare Cerebrium container."""
    required_modules = ("numpy", "pandas", "torch")
    if all(importlib.util.find_spec(name) is not None for name in required_modules):
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *RUNTIME_PACKAGES],
        check=True,
    )


def train(
    data_path: str = "data/sample/btcusdt_1s_sample.csv",
    max_ticks: int | str | None = None,
) -> dict:
    """Bootstrap dependencies and run the offline training cycle on Cerebrium."""
    _install_runtime_packages()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from bitcoin_predictor.train import run

    source = Path(data_path)
    if not source.is_absolute():
        source = root / source
    tick_limit = int(max_ticks) if max_ticks is not None else None
    return run(data_path=str(source), output="/persistent-storage/artifacts", max_ticks=tick_limit)


def health() -> dict:
    return {
        "status": "ok",
        "module": "train",
        "offline_runtime": True,
        "runtime_dependency_bootstrap": True,
    }
