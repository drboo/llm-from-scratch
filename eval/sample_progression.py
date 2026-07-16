"""
Day 13: Sample progression — generate text from each checkpoint in a directory
and display (or save) the progression so you can watch the model evolve from
noise → words → grammar over training.

Usage:
    # Print samples from every checkpoint:
    python eval/sample_progression.py --ckpt-dir checkpoints \
        --prompt "Once upon a time" --n-new 150

    # Save each sample to a file and print a summary table:
    python eval/sample_progression.py --ckpt-dir checkpoints \
        --prompt "Once upon a time" --save-dir checkpoints/samples

    # Show progression from already-saved sample files:
    python eval/sample_progression.py --samples-dir checkpoints/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT, ModelConfig
from inference.sample import sample as generate_sample


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: Path, n_head: int, ctx: int,
                device: torch.device) -> tuple[GPT, int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    embed_w = ckpt["model"]["embed.weight"]
    vocab_size, d_model = embed_w.shape
    n_layer = max(
        int(k.split(".")[1]) for k in ckpt["model"] if k.startswith("blocks.")
    ) + 1
    cfg   = ModelConfig(vocab_size=vocab_size, d_model=d_model,
                        n_head=n_head, n_layer=n_layer, ctx=ctx)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    step = ckpt.get("step", 0)
    return model, step


def _encode(prompt: str, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Byte-encode prompt; try Codec if available for BPE models."""
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if tok_path.exists() and vocab_size > 256:
        try:
            sys.path.insert(0, str(ROOT / "tokeniser"))
            from tokenizer import Codec  # type: ignore
            ids = Codec(str(tok_path)).encode(prompt)
            return torch.tensor([ids], dtype=torch.long, device=device)
        except Exception:
            pass
    ids = list(prompt.encode("utf-8"))
    return torch.tensor([ids], dtype=torch.long, device=device)


def _decode(ids: list[int], vocab_size: int) -> str:
    tok_path = ROOT / "tokeniser" / "tokenizer.json"
    if tok_path.exists() and vocab_size > 256:
        try:
            sys.path.insert(0, str(ROOT / "tokeniser"))
            from tokenizer import Codec  # type: ignore
            return Codec(str(tok_path)).decode(ids)
        except Exception:
            pass
    return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")


def generate_from_checkpoint(
    ckpt_path: Path,
    prompt: str,
    n_new: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    n_head: int = 6,
    ctx: int = 256,
    device: torch.device | None = None,
) -> tuple[int, str]:
    """
    Load checkpoint, generate text from prompt.

    Returns:
        (step, generated_text)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, step = _load_model(ckpt_path, n_head, ctx, device)
    prompt_ids  = _encode(prompt, model.cfg.vocab_size, device)
    out         = generate_sample(model, prompt_ids, n_new=n_new,
                                  temperature=temperature, top_k=top_k, top_p=top_p)
    new_ids     = out[0, prompt_ids.shape[1]:].tolist()
    text        = _decode(new_ids, model.cfg.vocab_size)
    return step, text


def sweep_checkpoints(
    ckpt_dir: Path,
    prompt: str,
    n_new: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    save_dir: Path | None = None,
    n_head: int = 6,
    ctx: int = 256,
) -> list[tuple[int, str]]:
    """
    Generate a sample from each .pt file in ckpt_dir (sorted by step).

    Returns list of (step, text) pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts  = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"No .pt files found in {ckpt_dir}")
        return []

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt_path in ckpts:
        step, text = generate_from_checkpoint(
            ckpt_path, prompt,
            n_new=n_new, temperature=temperature, top_k=top_k,
            n_head=n_head, ctx=ctx, device=device,
        )
        results.append((step, text))

        header = f"=== step {step:>6} | {ckpt_path.name} ==="
        print(header)
        print(text[:300].replace("\n", " "))
        print()

        if save_dir:
            out = save_dir / f"step_{step:06d}.txt"
            out.write_text(f"{header}\nPrompt: {prompt!r}\n\n{text}\n")

    return results


def show_saved_progression(samples_dir: Path) -> None:
    """Print a formatted view of previously saved sample files."""
    files = sorted(samples_dir.glob("step_*.txt"))
    if not files:
        print(f"No step_*.txt files found in {samples_dir}")
        return
    for f in files:
        print(f.read_text()[:500])
        print("-" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Sample progression across checkpoints (Day 13)")
    p.add_argument("--ckpt-dir",    default="",        help="directory of .pt checkpoints")
    p.add_argument("--samples-dir", default="",        help="show already-saved sample files")
    p.add_argument("--prompt",      default="Once upon a time")
    p.add_argument("--n-new",       type=int,   default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k",       type=int,   default=50)
    p.add_argument("--top-p",       type=float, default=1.0)
    p.add_argument("--save-dir",    default="",        help="save samples here as well as printing")
    p.add_argument("--n-head",      type=int,   default=6)
    p.add_argument("--ctx",         type=int,   default=256)
    args = p.parse_args()

    if args.samples_dir:
        show_saved_progression(Path(args.samples_dir))
    elif args.ckpt_dir:
        save_dir = Path(args.save_dir) if args.save_dir else None
        sweep_checkpoints(
            Path(args.ckpt_dir), args.prompt,
            n_new=args.n_new, temperature=args.temperature,
            top_k=args.top_k, save_dir=save_dir,
            n_head=args.n_head, ctx=args.ctx,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
