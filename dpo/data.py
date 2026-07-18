"""
Day 30: DPO preference dataset.

Provides hand-written preference pairs and an encoder that produces
(input_ids, labels) using the SFT chat template so DPO training
can use the same infrastructure as SFT.

Format on disk:
    dpo_train_chosen.bin   / dpo_val_chosen.bin   — uint16 token ids
    dpo_train_rejected.bin / dpo_val_rejected.bin  — uint16 token ids
    dpo_train_chosen_labels.bin  (int32, -100 on prompt)
    dpo_train_rejected_labels.bin
    dpo_train_offsets.npy  / dpo_val_offsets.npy  — uint32, length n+1
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class PreferencePair:
    prompt:   str
    chosen:   str
    rejected: str


# ---------------------------------------------------------------------------
# Hand-written preference pairs (40 total: 20 email, 20 code)
# ---------------------------------------------------------------------------

PREFERENCE_PAIRS: list[PreferencePair] = [
    # ── Email: professional vs. terse/rude ─────────────────────────────────
    PreferencePair(
        prompt="Write a professional email declining a Friday meeting invitation.",
        chosen=(
            "Thank you for the invitation. Unfortunately, I have a prior commitment "
            "on Friday that prevents me from attending. I would be happy to find "
            "another time that works for us both. Could we look at next week?"
        ),
        rejected="No I can't come Friday.",
    ),
    PreferencePair(
        prompt="Write an email to a colleague requesting feedback on a draft report.",
        chosen=(
            "Hi Sarah, I hope you are well. I have attached the draft report for "
            "your review. I would appreciate any feedback you have, particularly on "
            "the methodology section. Please let me know if you have questions."
        ),
        rejected="Look at my report and tell me what's wrong with it.",
    ),
    PreferencePair(
        prompt="Write an apology email for missing a deadline.",
        chosen=(
            "I sincerely apologize for missing yesterday's deadline. I underestimated "
            "the complexity of the task and should have communicated earlier. I have "
            "completed the work now and am submitting it immediately with a brief "
            "explanation of what caused the delay."
        ),
        rejected="Sorry I missed the deadline. Here is the thing.",
    ),
    PreferencePair(
        prompt="Write an email introducing yourself to a new team.",
        chosen=(
            "Hello everyone, I am excited to join the team as a software engineer "
            "starting this Monday. I have five years of experience in backend "
            "development and am looking forward to collaborating with all of you. "
            "Please feel free to reach out with any questions."
        ),
        rejected="Hi I am the new person starting Monday.",
    ),
    PreferencePair(
        prompt="Write a follow-up email after a job interview.",
        chosen=(
            "Dear Ms. Chen, thank you for taking the time to interview me yesterday "
            "for the senior engineer role. I enjoyed learning about the team's work "
            "on distributed systems and remain very interested in the position. "
            "Please do not hesitate to contact me if you need any additional "
            "information."
        ),
        rejected="Thanks for interviewing me. Hope I get the job.",
    ),
    PreferencePair(
        prompt="Write an email requesting a one-on-one meeting with your manager.",
        chosen=(
            "Hi Alex, I would like to schedule a brief one-on-one at your earliest "
            "convenience. I have a few updates on the current project and some "
            "questions about Q3 priorities. Would 30 minutes this week work for you?"
        ),
        rejected="Can we meet? I have stuff to discuss.",
    ),
    PreferencePair(
        prompt="Write an email to inform the team about a system outage.",
        chosen=(
            "Team, we are currently experiencing an outage affecting the production "
            "API. Our on-call engineers are investigating the root cause. We will "
            "provide updates every 30 minutes until the issue is resolved. We "
            "apologize for the inconvenience."
        ),
        rejected="The system is down. We are working on it.",
    ),
    PreferencePair(
        prompt="Write a thank-you email to a vendor after a successful project delivery.",
        chosen=(
            "Dear DataStream team, I wanted to take a moment to thank you for the "
            "exceptional work on the migration project. You delivered on time, "
            "communicated proactively throughout, and resolved issues quickly. "
            "We look forward to working with you again."
        ),
        rejected="Thanks for finishing the project.",
    ),
    PreferencePair(
        prompt="Write an email escalating a blocking issue to senior management.",
        chosen=(
            "Hi Lisa, I am escalating an issue that is blocking our launch. The "
            "payment provider integration has failed certification for the third "
            "time due to an API breaking change on their side. We need to either "
            "extend the launch date by two weeks or switch providers. I would "
            "appreciate a decision by EOD."
        ),
        rejected="We have a problem with the launch. The payment thing doesn't work.",
    ),
    PreferencePair(
        prompt="Write a polite email correcting a mistake in an invoice.",
        chosen=(
            "Dear Billing Team, I noticed that invoice #INV-2024-0391 lists a "
            "charge of $4,200 for five seats, whereas our contract specifies $750 "
            "per seat for a total of $3,750. Could you please issue a corrected "
            "invoice at your earliest convenience? Thank you."
        ),
        rejected="Your invoice is wrong. Please fix it.",
    ),
    PreferencePair(
        prompt="Write an email announcing a team member's promotion.",
        chosen=(
            "I am delighted to announce that Maria Garcia has been promoted to "
            "Principal Engineer effective next month. Maria has made outstanding "
            "contributions to our infrastructure reliability work and has been a "
            "tremendous mentor to junior engineers. Please join me in congratulating her."
        ),
        rejected="Maria is now a Principal Engineer. Congrats Maria.",
    ),
    PreferencePair(
        prompt="Write an email to a client explaining a project delay.",
        chosen=(
            "Dear Mr. Okafor, I am writing to inform you of a two-week delay in the "
            "delivery of Phase 2. We encountered unexpected complexity in the data "
            "migration step. We have added resources to the effort and are confident "
            "in the revised delivery date of March 14th. I apologize for the impact "
            "on your schedule."
        ),
        rejected="The project will be late by 2 weeks because of data migration problems.",
    ),
    PreferencePair(
        prompt="Write an out-of-office auto-reply email.",
        chosen=(
            "Thank you for your email. I am out of the office from December 23rd "
            "through January 2nd with limited access to email. For urgent matters, "
            "please contact my colleague James at james@company.com. I will respond "
            "to your message upon my return."
        ),
        rejected="I am out of office. Back Jan 2. Email James if urgent.",
    ),
    PreferencePair(
        prompt="Write an email requesting a budget increase for a project.",
        chosen=(
            "Hi David, I am writing to request a budget adjustment for the platform "
            "migration project. The initial estimate of $80k did not account for "
            "the required security audit, which has come in at $15k. I believe this "
            "is necessary to meet our compliance obligations. Could we discuss "
            "options at your convenience?"
        ),
        rejected="We need more money for the project. The security audit costs $15k extra.",
    ),
    PreferencePair(
        prompt="Write an email welcoming a new intern to the team.",
        chosen=(
            "Dear Jamie, welcome to the engineering team! We are thrilled to have "
            "you join us for the summer. Your onboarding schedule is attached. On "
            "your first day please check in with HR at 9am, then meet your mentor "
            "Chen at 10am. Do not hesitate to reach out if you have any questions "
            "before you start."
        ),
        rejected="Hi Jamie, welcome. Come to HR on Monday at 9am.",
    ),
    PreferencePair(
        prompt="Write an email to schedule a retrospective meeting.",
        chosen=(
            "Hi team, I would like to schedule our Sprint 14 retrospective. "
            "Please could you fill in your availability for next Thursday or Friday "
            "afternoon in the Doodle poll linked below? The meeting will run for "
            "60 minutes and we will use the Start/Stop/Continue format."
        ),
        rejected="Let's do a retro. When are people free next week?",
    ),
    PreferencePair(
        prompt="Write a professional email to request a reference letter.",
        chosen=(
            "Dear Professor Williams, I hope this message finds you well. I am "
            "applying for a software engineering position at Stripe and was "
            "wondering if you would be willing to provide a reference letter. "
            "I have attached my resume and the job description for context. "
            "The deadline is February 10th. Thank you for your consideration."
        ),
        rejected="Can you write me a reference? The deadline is Feb 10.",
    ),
    PreferencePair(
        prompt="Write an email to cancel a subscription service politely.",
        chosen=(
            "Dear Customer Success Team, I am writing to cancel my subscription "
            "to DataViz Pro, effective at the end of the current billing cycle. "
            "I have found it does not meet our team's current needs. Thank you "
            "for the service over the past year. Please confirm the cancellation "
            "at your earliest convenience."
        ),
        rejected="Please cancel my subscription. It doesn't work for us.",
    ),
    PreferencePair(
        prompt="Write an email proposing a new internal process improvement.",
        chosen=(
            "Hi team, I would like to propose we adopt a lightweight RFC process "
            "for significant technical decisions. Currently decisions are made "
            "inconsistently across teams, leading to duplication and misalignment. "
            "The RFC template would be a one-page document covering the problem, "
            "proposed solution, and trade-offs, reviewed asynchronously in two days. "
            "I am happy to pilot this with the next infrastructure decision."
        ),
        rejected="We should have a process for technical decisions. I can write a template.",
    ),
    PreferencePair(
        prompt="Write an email summarizing the outcome of a team meeting.",
        chosen=(
            "Hi all, here is a summary of today's architecture review. We decided "
            "to migrate the job scheduler from Celery to Temporal by end of Q2. "
            "Action items: Tom will draft the migration plan by Friday, Priya will "
            "survey existing Temporal usage internally, and I will book a follow-up "
            "for March 8th. Thanks for a productive discussion."
        ),
        rejected="We had a meeting today. We will use Temporal instead of Celery. Tom will write a plan.",
    ),

    # ── Code: correct vs. buggy/poor ───────────────────────────────────────
    PreferencePair(
        prompt="Write a Python function to compute the factorial of n.",
        chosen=(
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)"
        ),
        rejected=(
            "def factorial(n):\n"
            "    return n * factorial(n - 1)"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to check if a string is a palindrome.",
        chosen=(
            "def is_palindrome(s: str) -> bool:\n"
            "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return cleaned == cleaned[::-1]"
        ),
        rejected=(
            "def is_palindrome(s):\n"
            "    return s == s[::-1]"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to find the two numbers in a list that sum to a target.",
        chosen=(
            "def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:\n"
            "    seen: dict[int, int] = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        complement = target - n\n"
            "        if complement in seen:\n"
            "            return (seen[complement], i)\n"
            "        seen[n] = i\n"
            "    return None"
        ),
        rejected=(
            "def two_sum(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return i, j"
        ),
    ),
    PreferencePair(
        prompt="Write a Python context manager for timing a code block.",
        chosen=(
            "import time\n"
            "from contextlib import contextmanager\n\n"
            "@contextmanager\n"
            "def timer(label: str = ''):\n"
            "    start = time.perf_counter()\n"
            "    try:\n"
            "        yield\n"
            "    finally:\n"
            "        elapsed = time.perf_counter() - start\n"
            "        print(f'{label}: {elapsed:.4f}s')"
        ),
        rejected=(
            "import time\n\n"
            "def timer():\n"
            "    start = time.time()\n"
            "    yield\n"
            "    print(time.time() - start)"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to flatten a nested list.",
        chosen=(
            "def flatten(lst: list) -> list:\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        rejected=(
            "def flatten(lst):\n"
            "    return [x for x in lst]"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to merge two sorted lists.",
        chosen=(
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    result, i, j = [], 0, 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i]); i += 1\n"
            "        else:\n"
            "            result.append(b[j]); j += 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result"
        ),
        rejected=(
            "def merge_sorted(a, b):\n"
            "    return sorted(a + b)"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to count word frequencies in a string.",
        chosen=(
            "from collections import Counter\n\n"
            "def word_frequencies(text: str) -> dict[str, int]:\n"
            "    words = text.lower().split()\n"
            "    return dict(Counter(words))"
        ),
        rejected=(
            "def word_frequencies(text):\n"
            "    d = {}\n"
            "    for w in text.split():\n"
            "        d[w] = d[w] + 1\n"
            "    return d"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to validate an email address.",
        chosen=(
            "import re\n\n"
            "def is_valid_email(email: str) -> bool:\n"
            "    pattern = r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'\n"
            "    return bool(re.match(pattern, email))"
        ),
        rejected=(
            "def is_valid_email(email):\n"
            "    return '@' in email"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to implement binary search.",
        chosen=(
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1"
        ),
        rejected=(
            "def binary_search(arr, target):\n"
            "    for i, x in enumerate(arr):\n"
            "        if x == target:\n"
            "            return i\n"
            "    return -1"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to compute the running average of a list.",
        chosen=(
            "def running_average(nums: list[float]) -> list[float]:\n"
            "    result, total = [], 0.0\n"
            "    for i, n in enumerate(nums, start=1):\n"
            "        total += n\n"
            "        result.append(total / i)\n"
            "    return result"
        ),
        rejected=(
            "def running_average(nums):\n"
            "    return [sum(nums[:i+1]) / (i+1) for i in range(len(nums))]"
        ),
    ),
    PreferencePair(
        prompt="Write a Python class for a thread-safe counter.",
        chosen=(
            "import threading\n\n"
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self._value = 0\n"
            "        self._lock = threading.Lock()\n\n"
            "    def increment(self) -> None:\n"
            "        with self._lock:\n"
            "            self._value += 1\n\n"
            "    def value(self) -> int:\n"
            "        with self._lock:\n"
            "            return self._value"
        ),
        rejected=(
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.value = 0\n\n"
            "    def increment(self):\n"
            "        self.value += 1"
        ),
    ),
    PreferencePair(
        prompt="Write a Python generator to yield batches from a list.",
        chosen=(
            "from typing import Generator\n\n"
            "def batched(lst: list, batch_size: int) -> Generator[list, None, None]:\n"
            "    if batch_size <= 0:\n"
            "        raise ValueError('batch_size must be positive')\n"
            "    for i in range(0, len(lst), batch_size):\n"
            "        yield lst[i : i + batch_size]"
        ),
        rejected=(
            "def batched(lst, n):\n"
            "    batches = []\n"
            "    for i in range(0, len(lst), n):\n"
            "        batches.append(lst[i:i+n])\n"
            "    return batches"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to deep-copy a nested dict.",
        chosen=(
            "import copy\n\n"
            "def deep_copy_dict(d: dict) -> dict:\n"
            "    return copy.deepcopy(d)"
        ),
        rejected=(
            "def deep_copy_dict(d):\n"
            "    return dict(d)"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to retry a function call up to n times.",
        chosen=(
            "import time\n"
            "from typing import Callable, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "def retry(fn: Callable[[], T], n: int, delay: float = 0.5) -> T:\n"
            "    last_exc: Exception | None = None\n"
            "    for attempt in range(n):\n"
            "        try:\n"
            "            return fn()\n"
            "        except Exception as e:\n"
            "            last_exc = e\n"
            "            if attempt < n - 1:\n"
            "                time.sleep(delay)\n"
            "    raise RuntimeError(f'Failed after {n} attempts') from last_exc"
        ),
        rejected=(
            "def retry(fn, n):\n"
            "    for i in range(n):\n"
            "        fn()"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to parse a CSV line respecting quoted fields.",
        chosen=(
            "import csv\nimport io\n\n"
            "def parse_csv_line(line: str) -> list[str]:\n"
            "    return next(csv.reader(io.StringIO(line)))"
        ),
        rejected=(
            "def parse_csv_line(line):\n"
            "    return line.split(',')"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to chunk a string into pieces of length n.",
        chosen=(
            "def chunks(s: str, n: int) -> list[str]:\n"
            "    if n <= 0:\n"
            "        raise ValueError('n must be positive')\n"
            "    return [s[i : i + n] for i in range(0, len(s), n)]"
        ),
        rejected=(
            "def chunks(s, n):\n"
            "    return [s[i:i+n] for i in range(n)]"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to compute all prime numbers up to n.",
        chosen=(
            "def sieve(n: int) -> list[int]:\n"
            "    if n < 2:\n"
            "        return []\n"
            "    is_prime = [True] * (n + 1)\n"
            "    is_prime[0] = is_prime[1] = False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if is_prime[i]:\n"
            "            for j in range(i*i, n+1, i):\n"
            "                is_prime[j] = False\n"
            "    return [i for i, p in enumerate(is_prime) if p]"
        ),
        rejected=(
            "def sieve(n):\n"
            "    primes = []\n"
            "    for i in range(2, n):\n"
            "        for j in range(2, i):\n"
            "            if i % j == 0:\n"
            "                break\n"
            "        else:\n"
            "            primes.append(i)\n"
            "    return primes"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to group a list of dicts by a key.",
        chosen=(
            "from collections import defaultdict\n\n"
            "def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:\n"
            "    groups: dict[str, list] = defaultdict(list)\n"
            "    for item in items:\n"
            "        groups[item[key]].append(item)\n"
            "    return dict(groups)"
        ),
        rejected=(
            "def group_by(items, key):\n"
            "    d = {}\n"
            "    for item in items:\n"
            "        k = item[key]\n"
            "        if k not in d:\n"
            "            d[k] = []\n"
            "        d[k].append(item)\n"
            "    return d"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function that returns the most common element in a list.",
        chosen=(
            "from collections import Counter\n\n"
            "def most_common(lst: list) -> object:\n"
            "    if not lst:\n"
            "        raise ValueError('list is empty')\n"
            "    return Counter(lst).most_common(1)[0][0]"
        ),
        rejected=(
            "def most_common(lst):\n"
            "    return max(lst)"
        ),
    ),
    PreferencePair(
        prompt="Write a Python function to rotate a list by k positions.",
        chosen=(
            "def rotate(lst: list, k: int) -> list:\n"
            "    if not lst:\n"
            "        return lst\n"
            "    k = k % len(lst)\n"
            "    return lst[-k:] + lst[:-k] if k else lst[:]"
        ),
        rejected=(
            "def rotate(lst, k):\n"
            "    for _ in range(k):\n"
            "        lst.append(lst.pop(0))\n"
            "    return lst"
        ),
    ),
]

# 90/10 train/val split
_N_VAL = max(1, len(PREFERENCE_PAIRS) // 10)
TRAIN_PAIRS = PREFERENCE_PAIRS[_N_VAL:]
VAL_PAIRS   = PREFERENCE_PAIRS[:_N_VAL]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _chat_head(prompt: str) -> str:
    return f"<|bos|><|user|>{prompt}<|endofturn|><|assistant|>"


def encode_preference_pair(
    pair:       PreferencePair,
    codec,
    max_tokens: int = 512,
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]] | None:
    """
    Encode one preference pair using the chat template.

    Returns ((chosen_ids, chosen_labels), (rejected_ids, rejected_labels))
    or None if either sequence exceeds max_tokens.

    Labels are -100 on prompt tokens so only response tokens
    contribute to the DPO log-probability computation.
    """
    def _encode_one(response: str):
        head = _chat_head(pair.prompt)
        tail = response + "<|endofturn|><|eos|>"
        if codec is not None:
            head_ids = codec.encode(head)
            tail_ids = codec.encode(tail)
        else:
            head_ids = list(head.encode("utf-8"))
            tail_ids = list(tail.encode("utf-8"))
        if len(head_ids) + len(tail_ids) > max_tokens:
            return None
        ids    = head_ids + tail_ids
        labels = [-100] * len(head_ids) + tail_ids
        return ids, labels

    chosen   = _encode_one(pair.chosen)
    rejected = _encode_one(pair.rejected)
    if chosen is None or rejected is None:
        return None
    return chosen, rejected


def _write_split(
    pairs:   list[PreferencePair],
    out_dir: Path,
    split:   str,
    codec,
    max_tokens: int = 512,
) -> dict:
    """Encode and write one split to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen_ids_list:    list[list[int]] = []
    chosen_labels_list: list[list[int]] = []
    rej_ids_list:       list[list[int]] = []
    rej_labels_list:    list[list[int]] = []
    skipped = 0

    for pair in pairs:
        result = encode_preference_pair(pair, codec, max_tokens)
        if result is None:
            skipped += 1
            continue
        (c_ids, c_lab), (r_ids, r_lab) = result
        chosen_ids_list.append(c_ids)
        chosen_labels_list.append(c_lab)
        rej_ids_list.append(r_ids)
        rej_labels_list.append(r_lab)

    n = len(chosen_ids_list)

    def _write_ids(sequences, fname):
        offsets = np.zeros(len(sequences) + 1, dtype=np.uint32)
        for i, s in enumerate(sequences):
            offsets[i + 1] = offsets[i] + len(s)
        flat = np.array([t for s in sequences for t in s], dtype=np.uint16)
        np.save(str(out_dir / f"{fname}_offsets.npy"), offsets)
        flat.tofile(str(out_dir / f"{fname}.bin"))

    def _write_labels(sequences, fname):
        offsets = np.zeros(len(sequences) + 1, dtype=np.uint32)
        for i, s in enumerate(sequences):
            offsets[i + 1] = offsets[i] + len(s)
        flat = np.array([t for s in sequences for t in s], dtype=np.int32)
        np.save(str(out_dir / f"{fname}_offsets.npy"), offsets)
        flat.tofile(str(out_dir / f"{fname}.bin"))

    _write_ids(chosen_ids_list, f"dpo_{split}_chosen")
    _write_ids(rej_ids_list,    f"dpo_{split}_rejected")
    _write_labels(chosen_labels_list, f"dpo_{split}_chosen_labels")
    _write_labels(rej_labels_list,    f"dpo_{split}_rejected_labels")

    return {"split": split, "n": n, "skipped": skipped}


def build_dpo_dataset(
    out_dir:    str | Path,
    codec       = None,
    max_tokens: int = 512,
) -> dict:
    """Build train + val splits from PREFERENCE_PAIRS."""
    out_dir = Path(out_dir)
    stats = {}
    stats["train"] = _write_split(TRAIN_PAIRS, out_dir, "train", codec, max_tokens)
    stats["val"]   = _write_split(VAL_PAIRS,   out_dir, "val",   codec, max_tokens)
    stats["total_pairs"] = len(PREFERENCE_PAIRS)
    return stats


# ---------------------------------------------------------------------------
# In-memory batch loader (used by train/dpo.py)
# ---------------------------------------------------------------------------


def load_split_inmemory(
    pairs:      list[PreferencePair],
    codec,
    max_tokens: int = 512,
) -> list[tuple]:
    """Return list of ((c_ids, c_lab), (r_ids, r_lab)) for all valid pairs."""
    out = []
    for pair in pairs:
        result = encode_preference_pair(pair, codec, max_tokens)
        if result is not None:
            out.append(result)
    return out
