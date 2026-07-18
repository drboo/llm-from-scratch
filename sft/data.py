"""
Day 22: SFT dataset — instruction data formatted in the chat template.

Sources (all optional, each can be disabled via --n-X 0):
  • tatsu-lab/alpaca          general instruction following
  • sahil2801/CodeAlpaca-20k  code-focused instructions
  • OpenAssistant/oasst2      human/assistant conversation pairs (English)
  • HAND_WRITTEN              30 curated email + Python examples

On-disk format (three parallel files):
  sft_train.bin     uint16  flat token ids, examples packed end-to-end
  sft_labels.bin    int32   parallel labels; -100 = no loss, else = token id
  sft_offsets.npy   uint32  start positions; len = n_examples + 1 (sentinel)
  sft_stats.json            source counts, token stats, filter reasons

Usage:
    python sft/data.py --out data/sft
    python sft/data.py --out data/sft --n-alpaca 5000 --n-code 2000 --n-oasst 1000
    python sft/data.py --out data/sft --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tokeniser"))

TOKENIZER_PATH = ROOT / "tokeniser" / "tokenizer.json"

# ---------------------------------------------------------------------------
# Junk filters
# ---------------------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"^(i('m| am) (sorry|unable|not able)|i cannot|i can't|as an ai|"
    r"i don't have (the ability|access)|it (is|would be) (not |in)appropriate)",
    re.IGNORECASE,
)

_MIN_RESPONSE_CHARS = 20
_MAX_EXAMPLE_TOKENS = 512


def is_junk(instruction: str, response: str) -> tuple[bool, str]:
    """Return (True, reason) if the example should be dropped."""
    if not instruction or not instruction.strip():
        return True, "empty_instruction"
    if not response or not response.strip():
        return True, "empty_response"
    if len(response.strip()) < _MIN_RESPONSE_CHARS:
        return True, "short_response"
    if _REFUSAL_RE.match(response.strip()):
        return True, "refusal"
    return False, ""


# ---------------------------------------------------------------------------
# Hand-written examples (email + Python)
# ---------------------------------------------------------------------------

HAND_WRITTEN: list[dict] = [
    # ── email examples ────────────────────────────────────────────────────
    {
        "instruction": "Write a short email declining a Friday afternoon meeting.",
        "response": (
            "Subject: Re: Friday Afternoon Meeting\n\n"
            "Hi,\n\nThank you for the invitation. Unfortunately I have a prior "
            "commitment on Friday afternoon and won't be able to attend. Would it "
            "be possible to reschedule for early next week? I'm free Monday or "
            "Tuesday morning.\n\nBest regards"
        ),
    },
    {
        "instruction": "Write a professional email requesting a project status update.",
        "response": (
            "Subject: Project Status Update Request\n\n"
            "Hi,\n\nI wanted to check in on the progress of the project. Could you "
            "please share a brief status update, including any blockers you're "
            "facing and the expected timeline for the next milestone?\n\n"
            "Thank you for keeping me informed.\n\nBest regards"
        ),
    },
    {
        "instruction": "Write an email introducing a new team member to the rest of the team.",
        "response": (
            "Subject: Welcome Our New Team Member\n\n"
            "Hi everyone,\n\nI'm delighted to introduce our newest team member, who "
            "joins us this Monday. They bring extensive experience in their field "
            "and will be contributing to our upcoming projects. Please join me in "
            "giving them a warm welcome!\n\nFeel free to reach out and introduce "
            "yourselves.\n\nBest"
        ),
    },
    {
        "instruction": "Write a polite follow-up email for an unanswered message sent a week ago.",
        "response": (
            "Subject: Following Up — [Original Subject]\n\n"
            "Hi,\n\nI hope you're doing well. I wanted to follow up on my email "
            "from last week regarding [topic]. I understand you're likely busy, "
            "but I'd appreciate any update when you have a moment.\n\n"
            "Please let me know if you need any additional information from my "
            "end.\n\nThank you for your time."
        ),
    },
    {
        "instruction": "Write an email to schedule a one-hour meeting with a client next week.",
        "response": (
            "Subject: Meeting Request — Next Week\n\n"
            "Dear [Client Name],\n\nI hope this email finds you well. I'd like to "
            "schedule a one-hour meeting to discuss [topic]. Would any of the "
            "following times work for you?\n\n"
            "  • Monday, [date] at 10:00 AM\n"
            "  • Tuesday, [date] at 2:00 PM\n"
            "  • Wednesday, [date] at 11:00 AM\n\n"
            "Please let me know your preference or suggest an alternative time. "
            "I'll send a calendar invite once we confirm.\n\nBest regards"
        ),
    },
    {
        "instruction": "Write a thank-you email to a colleague who helped you resolve a production issue.",
        "response": (
            "Subject: Thank You for Your Help!\n\n"
            "Hi,\n\nI just wanted to take a moment to thank you for your help "
            "resolving the production issue yesterday. Your quick thinking and "
            "expertise made all the difference — we got the system back online "
            "much faster than expected.\n\nI really appreciate you dropping "
            "everything to help out. It means a lot.\n\nThanks again!"
        ),
    },
    {
        "instruction": "Write a brief apology email for a delayed response.",
        "response": (
            "Subject: Apologies for the Delayed Response\n\n"
            "Hi,\n\nI apologize for the delay in getting back to you. I've been "
            "managing a high volume of tasks this week and your message got buried. "
            "That's no excuse, and I should have responded sooner.\n\n"
            "To answer your question: [answer here].\n\n"
            "Again, I'm sorry for the wait and will be more prompt going forward."
        ),
    },
    {
        "instruction": "Write an email requesting a two-week deadline extension on a deliverable.",
        "response": (
            "Subject: Request for Deadline Extension\n\n"
            "Hi,\n\nI'm writing to request a two-week extension on the [deliverable] "
            "originally due [date]. We've encountered [brief reason] which has "
            "impacted our timeline. With an extension to [new date], I'm confident "
            "we can deliver a thorough, high-quality result.\n\n"
            "I understand this may affect downstream plans and I'm happy to discuss "
            "mitigation options if needed.\n\nThank you for your consideration."
        ),
    },
    {
        "instruction": "Write an email announcing the successful completion of a project.",
        "response": (
            "Subject: Project [Name] — Completed Successfully!\n\n"
            "Hi team,\n\nI'm thrilled to announce that Project [Name] has been "
            "completed on time and within budget. This was a significant effort and "
            "every member of the team contributed to its success.\n\n"
            "Key achievements:\n  • [Achievement 1]\n  • [Achievement 2]\n\n"
            "Thank you all for your dedication and hard work. A full retrospective "
            "will be scheduled for next week.\n\nCongratulations!"
        ),
    },
    {
        "instruction": "Write a welcome email to a new client starting an engagement.",
        "response": (
            "Subject: Welcome — We're Excited to Work with You!\n\n"
            "Dear [Client Name],\n\nWelcome! We're delighted to have you on board "
            "and look forward to a productive partnership.\n\n"
            "As a next step, I'll schedule a kickoff call to align on goals, "
            "timelines, and communication preferences. In the meantime, please "
            "don't hesitate to reach out if you have any questions.\n\n"
            "We're committed to your success and can't wait to get started.\n\n"
            "Warm regards"
        ),
    },
    {
        "instruction": "Write an email requesting feedback on a document you've shared.",
        "response": (
            "Subject: Feedback Request — [Document Name]\n\n"
            "Hi,\n\nI've attached [document name] for your review. I'd really "
            "value your feedback, particularly on [specific sections or aspects].\n\n"
            "If possible, could you share your thoughts by [date]? Even brief "
            "comments would be extremely helpful.\n\nThank you in advance!"
        ),
    },
    {
        "instruction": "Write a reminder email about an upcoming project deadline.",
        "response": (
            "Subject: Reminder: [Project] Deadline on [Date]\n\n"
            "Hi team,\n\nThis is a friendly reminder that the deadline for "
            "[deliverable] is [date] — just [X] days away.\n\n"
            "Please ensure all outstanding tasks are complete and flag any blockers "
            "as soon as possible so we have time to address them.\n\n"
            "Thanks for staying on top of this!"
        ),
    },
    {
        "instruction": "Write a professional response to a client complaint about slow service.",
        "response": (
            "Subject: Re: Your Recent Experience\n\n"
            "Dear [Client Name],\n\nThank you for bringing this to our attention. "
            "I sincerely apologize for the delays you experienced — this falls "
            "below the standard of service we strive to provide.\n\n"
            "I've investigated the issue and [brief explanation]. We've taken "
            "steps to prevent this from happening again.\n\nAs a gesture of good "
            "faith, [compensation or remedy if applicable]. Please feel free to "
            "contact me directly if you have further concerns.\n\nSincerely"
        ),
    },
    {
        "instruction": "Write a casual email inviting the team to a Friday lunch.",
        "response": (
            "Subject: Team Lunch This Friday!\n\n"
            "Hey everyone,\n\nQuick note — I'm organizing a team lunch this Friday "
            "at [restaurant/location] at 12:30 PM. It'll be a great chance to "
            "unwind and catch up outside of meetings.\n\n"
            "Let me know by Wednesday if you can make it so I can book the right "
            "number of seats.\n\nHope to see you all there!"
        ),
    },
    {
        "instruction": "Write an email notifying users about scheduled system maintenance.",
        "response": (
            "Subject: Scheduled System Maintenance — [Date]\n\n"
            "Dear users,\n\nWe will be performing scheduled maintenance on "
            "[system name] on [date] from [start time] to [end time] "
            "[timezone]. During this window the system will be unavailable.\n\n"
            "We recommend saving your work beforehand and planning accordingly. "
            "We apologize for any inconvenience and will notify you as soon as "
            "service has been restored.\n\nThank you for your understanding."
        ),
    },
    # ── Python examples ────────────────────────────────────────────────────
    {
        "instruction": "Write a Python function that returns the nth Fibonacci number.",
        "response": (
            "```python\n"
            "def fibonacci(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError(\"n must be non-negative\")\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "response": (
            "```python\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    cleaned = s.lower().replace(\" \", \"\")\n"
            "    return cleaned == cleaned[::-1]\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to count word frequencies in a string.",
        "response": (
            "```python\n"
            "def word_frequencies(text: str) -> dict[str, int]:\n"
            "    counts: dict[str, int] = {}\n"
            "    for word in text.lower().split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function implementing binary search.",
        "response": (
            "```python\n"
            "def binary_search(arr: list, target) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to check if a number is prime.",
        "response": (
            "```python\n"
            "def is_prime(n: int) -> bool:\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n == 2:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    for i in range(3, int(n ** 0.5) + 1, 2):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to flatten a nested list.",
        "response": (
            "```python\n"
            "def flatten(nested: list) -> list:\n"
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to merge two sorted lists into one sorted list.",
        "response": (
            "```python\n"
            "def merge_sorted(a: list, b: list) -> list:\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python decorator that measures and prints a function's execution time.",
        "response": (
            "```python\n"
            "import time\n"
            "import functools\n\n"
            "def timed(func):\n"
            "    @functools.wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        start = time.perf_counter()\n"
            "        result = func(*args, **kwargs)\n"
            "        elapsed = time.perf_counter() - start\n"
            "        print(f\"{func.__name__} took {elapsed:.4f}s\")\n"
            "        return result\n"
            "    return wrapper\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python class implementing a stack with push, pop, and peek.",
        "response": (
            "```python\n"
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._data = []\n\n"
            "    def push(self, item) -> None:\n"
            "        self._data.append(item)\n\n"
            "    def pop(self):\n"
            "        if not self._data:\n"
            "            raise IndexError(\"pop from empty stack\")\n"
            "        return self._data.pop()\n\n"
            "    def peek(self):\n"
            "        if not self._data:\n"
            "            raise IndexError(\"peek at empty stack\")\n"
            "        return self._data[-1]\n\n"
            "    def __len__(self) -> int:\n"
            "        return len(self._data)\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to validate an email address using regex.",
        "response": (
            "```python\n"
            "import re\n\n"
            "_EMAIL_RE = re.compile(\n"
            "    r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'\n"
            ")\n\n"
            "def is_valid_email(address: str) -> bool:\n"
            "    return bool(_EMAIL_RE.match(address))\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to compute a running average from a list of numbers.",
        "response": (
            "```python\n"
            "def running_average(numbers: list[float]) -> list[float]:\n"
            "    averages = []\n"
            "    total = 0.0\n"
            "    for i, n in enumerate(numbers, start=1):\n"
            "        total += n\n"
            "        averages.append(total / i)\n"
            "    return averages\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to read a CSV file and return a list of dicts.",
        "response": (
            "```python\n"
            "import csv\n\n"
            "def read_csv(path: str) -> list[dict]:\n"
            "    with open(path, newline='', encoding='utf-8') as f:\n"
            "        reader = csv.DictReader(f)\n"
            "        return list(reader)\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to find all duplicates in a list.",
        "response": (
            "```python\n"
            "def find_duplicates(items: list) -> list:\n"
            "    seen = set()\n"
            "    duplicates = set()\n"
            "    for item in items:\n"
            "        if item in seen:\n"
            "            duplicates.add(item)\n"
            "        seen.add(item)\n"
            "    return list(duplicates)\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python context manager that temporarily changes the working directory.",
        "response": (
            "```python\n"
            "import os\n"
            "from contextlib import contextmanager\n\n"
            "@contextmanager\n"
            "def change_dir(path: str):\n"
            "    original = os.getcwd()\n"
            "    os.chdir(path)\n"
            "    try:\n"
            "        yield\n"
            "    finally:\n"
            "        os.chdir(original)\n"
            "```"
        ),
    },
    {
        "instruction": "Write a Python function to chunk a list into fixed-size pieces.",
        "response": (
            "```python\n"
            "def chunks(lst: list, size: int):\n"
            "    for i in range(0, len(lst), size):\n"
            "        yield lst[i : i + size]\n"
            "```"
        ),
    },
]


# ---------------------------------------------------------------------------
# Dataset iterators
# ---------------------------------------------------------------------------


def _alpaca_to_example(ex: dict) -> dict:
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    if inp:
        instruction = f"{instruction}\n\nInput: {inp}"
    return {"instruction": instruction, "response": ex.get("output", "")}


def iter_alpaca(n: int) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    count = 0
    for ex in ds:
        if count >= n:
            break
        yield _alpaca_to_example(ex)
        count += 1


def iter_code_alpaca(n: int) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
    count = 0
    for ex in ds:
        if count >= n:
            break
        yield _alpaca_to_example(ex)
        count += 1


def iter_oasst(n: int) -> Iterator[dict]:
    """Yield English (human, assistant) pairs from OpenAssistant/oasst2."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    # Index prompter messages and pair with their best assistant reply
    # We do a single-pass: collect prompter messages and their parent_id=None
    # (top-level), then on the next message look for matching assistant replies.
    # Simpler approach: just yield adjacent prompter→assistant pairs by rank=0.
    pending: dict[str, str] = {}  # message_id -> text for prompter messages
    count = 0
    for ex in ds:
        if count >= n:
            break
        if ex.get("lang") != "en":
            continue
        role = ex.get("role", "")
        mid = ex.get("message_id", "")
        pid = ex.get("parent_id")
        text = ex.get("text", "")
        if role == "prompter":
            pending[mid] = text
        elif role == "assistant" and pid in pending:
            yield {"instruction": pending.pop(pid), "response": text}
            count += 1


# ---------------------------------------------------------------------------
# Encoding with loss mask
# ---------------------------------------------------------------------------


def encode_example(
    instruction: str,
    response: str,
    codec,
    max_tokens: int = _MAX_EXAMPLE_TOKENS,
) -> tuple[list[int], list[int]] | None:
    """
    Encode one example via codec.encode_chat.

    Returns (ids, labels) where labels[i] = ids[i] on assistant tokens
    and -100 elsewhere.  Returns None if the encoded length exceeds max_tokens.
    """
    ids, mask = codec.encode_chat(instruction, response)
    if len(ids) > max_tokens:
        return None
    labels = [t if m else -100 for t, m in zip(ids, mask)]
    return ids, labels


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_sft_dataset(
    out_dir: Path,
    n_alpaca: int = 20_000,
    n_code: int = 10_000,
    n_oasst: int = 5_000,
    max_tokens: int = _MAX_EXAMPLE_TOKENS,
    dry_run: bool = False,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Build the SFT dataset from all sources and write to out_dir.

    Returns a stats dict.
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run python tokeniser/tokeniser.py first."
        )
    from tokenizer import Codec  # type: ignore
    codec = Codec(str(TOKENIZER_PATH))

    filter_counts: dict[str, int] = {}
    all_examples: list[tuple[list[int], list[int]]] = []
    src_counts: dict[str, int] = {}

    def _process_source(name: str, iterator: Iterator[dict], limit: int) -> None:
        accepted = dropped = 0
        reasons: dict[str, int] = {}
        print(f"\n[{name}] encoding up to {limit:,} examples …")
        for ex in iterator:
            if accepted + dropped >= limit * 3:
                break
            bad, reason = is_junk(ex.get("instruction", ""), ex.get("response", ""))
            if bad:
                reasons[reason] = reasons.get(reason, 0) + 1
                dropped += 1
                continue
            result = encode_example(ex["instruction"], ex["response"], codec, max_tokens)
            if result is None:
                reasons["too_long"] = reasons.get("too_long", 0) + 1
                dropped += 1
                continue
            all_examples.append(result)
            accepted += 1
            if accepted >= limit:
                break
        print(f"  accepted {accepted:,}  dropped {dropped:,}  reasons: {reasons}")
        filter_counts[name] = reasons
        src_counts[name] = accepted

    # hand-written first (always included)
    _process_source(
        "hand_written",
        iter(HAND_WRITTEN),
        len(HAND_WRITTEN),
    )

    if n_alpaca > 0:
        _process_source("alpaca", iter_alpaca(n_alpaca * 3), n_alpaca)
    if n_code > 0:
        _process_source("code_alpaca", iter_code_alpaca(n_code * 3), n_code)
    if n_oasst > 0:
        _process_source("oasst", iter_oasst(n_oasst * 5), n_oasst)

    print(f"\nTotal examples before split: {len(all_examples):,}")

    if dry_run:
        print("[dry-run] Skipping file write.")
        return {"total": len(all_examples), "src_counts": src_counts, "dry_run": True}

    # Shuffle and split
    import random
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_fraction))
    val_examples   = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    print(f"  train: {len(train_examples):,}  val: {len(val_examples):,}")

    out_dir = Path(out_dir)
    _write_split(train_examples, out_dir, "train")
    _write_split(val_examples,   out_dir, "val")

    stats = {
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "src_counts": src_counts,
        "filter_reasons": filter_counts,
        "max_tokens_per_example": max_tokens,
        "val_fraction": val_fraction,
    }
    (out_dir / "sft_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {out_dir / 'sft_stats.json'}")
    return stats


def _write_split(
    examples: list[tuple[list[int], list[int]]],
    out_dir: Path,
    split: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ids: list[int] = []
    all_labels: list[int] = []
    offsets: list[int] = [0]
    for ids, labels in examples:
        all_ids.extend(ids)
        all_labels.extend(labels)
        offsets.append(len(all_ids))

    tokens_arr = np.array(all_ids,    dtype=np.uint16)
    labels_arr = np.array(all_labels, dtype=np.int32)
    offsets_arr = np.array(offsets,   dtype=np.uint32)

    token_path  = out_dir / f"sft_{split}.bin"
    labels_path = out_dir / f"sft_{split}_labels.bin"
    offsets_path = out_dir / f"sft_{split}_offsets.npy"

    fp = np.memmap(str(token_path), dtype=np.uint16, mode="w+", shape=(len(tokens_arr),))
    fp[:] = tokens_arr; fp.flush(); del fp

    lp = np.memmap(str(labels_path), dtype=np.int32, mode="w+", shape=(len(labels_arr),))
    lp[:] = labels_arr; lp.flush(); del lp

    np.save(str(offsets_path), offsets_arr)

    n_loss = int((labels_arr >= 0).sum())
    print(
        f"  [{split}] {len(examples):,} examples  "
        f"{len(all_ids)/1e3:.1f}k tokens  "
        f"{n_loss/len(all_ids)*100:.1f}% carry loss  "
        f"→ {token_path.name}, {labels_path.name}, {offsets_path.name}"
    )


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def verify_example(
    out_dir: Path,
    split: str = "train",
    idx: int = 0,
) -> None:
    """Print one encoded example showing which tokens carry loss."""
    from tokenizer import Codec  # type: ignore
    codec = Codec(str(TOKENIZER_PATH))

    out_dir = Path(out_dir)
    offsets = np.load(str(out_dir / f"sft_{split}_offsets.npy"))
    tokens  = np.memmap(str(out_dir / f"sft_{split}.bin"),        dtype=np.uint16, mode="r")
    labels  = np.memmap(str(out_dir / f"sft_{split}_labels.bin"), dtype=np.int32,  mode="r")

    start, end = int(offsets[idx]), int(offsets[idx + 1])
    ids  = list(map(int, tokens[start:end]))
    labs = list(map(int, labels[start:end]))

    print(f"\n{'='*60}")
    print(f"Example {idx}  ({end-start} tokens)")
    print("=" * 60)
    print("TOKEN".ljust(20), "ID".rjust(6), "LOSS")
    for i, (tok_id, lab) in enumerate(zip(ids, labs)):
        tok_str = repr(codec.decode([tok_id]))[:18]
        loss_marker = "YES" if lab >= 0 else " — "
        print(f"  {tok_str:<20} {tok_id:>6}  {loss_marker}")
        if i >= 40:
            print(f"  ... ({end-start-41} more tokens)")
            break
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT dataset (Day 22)")
    p.add_argument("--out",       type=Path, default=Path("data/sft"))
    p.add_argument("--n-alpaca",  type=int,  default=20_000)
    p.add_argument("--n-code",    type=int,  default=10_000)
    p.add_argument("--n-oasst",   type=int,  default=5_000)
    p.add_argument("--max-tokens", type=int, default=_MAX_EXAMPLE_TOKENS)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--verify",    action="store_true",
                   help="Print one decoded example after building")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    stats = build_sft_dataset(
        out_dir      = args.out,
        n_alpaca     = args.n_alpaca,
        n_code       = args.n_code,
        n_oasst      = args.n_oasst,
        max_tokens   = args.max_tokens,
        dry_run      = args.dry_run,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
    if args.verify and not args.dry_run:
        verify_example(args.out, "train", idx=0)
