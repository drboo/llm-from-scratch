"""
Checkpoint save / load — model + optimizer + step + RNG state.

Saving the full RNG state means a resumed run is byte-for-byte identical
to an uninterrupted one (assuming deterministic data sampling).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
) -> None:
    """Serialize training state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "step":            step,
            "loss":            loss,
            "rng_cpu":         torch.get_rng_state(),
            "rng_numpy":       np.random.get_state(),
            "rng_cuda":        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """Restore training state from disk.

    Returns:
        (step, loss) — the values at the time of the save.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    torch.set_rng_state(ckpt["rng_cpu"])
    np.random.set_state(ckpt["rng_numpy"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    return ckpt["step"], ckpt["loss"]
