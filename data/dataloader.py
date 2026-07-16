"""
Day 8: memmap batch sampler.

get_batch picks random offsets into a flat uint16 .bin file and returns
(x, y) GPU tensors.  No padding — sequences are packed end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_batch(
    split: str,
    data_dir: str | Path,
    ctx: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a memmap .bin file.

    The file is a flat array of uint16 token ids written by prepare_toy.py.
    We pick `batch_size` random start offsets, then read two overlapping
    windows of length `ctx`:
        x = data[i : i+ctx]        — input tokens
        y = data[i+1 : i+ctx+1]    — targets (x shifted left by one)

    Args:
        split:     "train" or "val"
        data_dir:  directory containing {split}.bin
        ctx:       context length T
        batch_size: sequences per batch B
        device:    target device
        generator: optional RNG for reproducibility

    Returns:
        x: (B, T) int64
        y: (B, T) int64
    """
    path = Path(data_dir) / f"{split}.bin"
    data = np.memmap(str(path), dtype=np.uint16, mode="r")

    # Random start positions — ensure the window i+1..i+ctx+1 stays in bounds
    ix = torch.randint(len(data) - ctx, (batch_size,), generator=generator)

    x = torch.stack([torch.from_numpy(data[i    : i + ctx    ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + ctx + 1].astype(np.int64)) for i in ix])

    device = torch.device(device)
    if device.type == "cuda":
        # Async transfer to GPU; pin_memory avoids an extra copy
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y
