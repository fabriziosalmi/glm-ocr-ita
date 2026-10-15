"""Scoring primitives. Pure functions, no model and no file access: everything here is covered by tests/test_metrics.py.

Quality is three numbers (rule 5 of docs/ROADMAP.md): entity recall, pages with a skipped block, repetition loops.
"""
import collections
import difflib
import random
import re
import unicodedata
import zlib

MAX_TOKENS = 3000        # token budget of every run in results/
LOOP_RATIO = 0.25        # zlib ratio below which a text is a repetition loop
LOOP_MIN_BYTES = 200     # shorter texts compress badly whatever they hold
SKIP_RATIO = 0.90        # an upright page with fewer characters than this share of the reference skipped a block
BOOT_N = 2000


def squash(s):
    """NFKC, lowercase, no whitespace: the form entities are matched in."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).lower())


def alnum(s):
    """Letters, digits and currency signs only: the form CER is computed on, blind to markup and spacing."""
    return re.sub(r"[^0-9a-zà-ÿ€$%]", "", unicodedata.normalize("NFKC", s).lower())


def found(entities, text):
    """How many entities come back exactly. Recall only: a short entity can also match by accident."""
    h = squash(text)
    return sum(squash(e["text"]) in h for e in entities)


def flip_notation(s):
    """2.100.000,00 <-> 2,100,000.00"""
    return s.translate(str.maketrans(".,", ",."))


def notation_flips(entities, text):
    """(flipped, candidates): entities holding both a dot and a comma between digits that are missing as written
    but present with the two swapped. The base model does this to Italian amounts."""
    h = squash(text)
    cand = [squash(e["text"]) for e in entities if re.search(r"\d[.,]\d.*\d[.,]\d", e["text"]) and "." in e["text"] and "," in e["text"]]
    return sum(c not in h and flip_notation(c) in h for c in cand), len(cand)


ACCENTED = re.compile(r"[àèéìíòóùú]")


def accent_recall(reference, text):
    """(found, total) over the words of the reference that hold an accented letter, each looked up as written."""
    h = squash(text)
    words = [w for w in (squash(w).strip(".,;:()[]\"'«»") for w in reference.split()) if ACCENTED.search(w)]
    return sum(w in h for w in words), len(words)


def compress_ratio(text):
    raw = text.encode()
    return len(zlib.compress(raw, 9)) / len(raw) if raw else 1.0


def is_loop(text, gen_tokens, max_tokens=MAX_TOKENS):
    """The decoder ran out of tokens, or the text compresses far better than a real page does."""
    return gen_tokens >= max_tokens or (len(text.encode()) >= LOOP_MIN_BYTES and compress_ratio(text) < LOOP_RATIO)


def loop_start(text, max_period=400, min_repeats=4, min_span=160):
    """(index, period) of a repetition the text ends with, or None: one unit of at most max_period characters, repeated
    at least min_repeats times in a row over at least min_span characters, right up to the last character.
    Used while a page is being read, to stop a loop as it starts instead of paying the whole token budget for it.
    On the 4201 outputs saved by 2026-10-14, checked every 200 characters: 68 of 75 loops caught, after a fifth of
    their length in the median; it fired on one output that is_loop does not flag, a sentence written five times."""
    n = len(text)
    for p in range(1, min(max_period, n // min_repeats) + 1):
        unit = text[n - p:]
        if text[n - 2 * p:n - p] != unit:
            continue
        k = 2
        while n - (k + 1) * p >= 0 and text[n - (k + 1) * p:n - k * p] == unit:
            k += 1
        if k >= min_repeats and k * p >= min_span:
            return n - k * p, p
    return None


# Checks of a training answer against the teacher's reading of the same page (experiments/ft_targets.py). Letters and digits
# only, so that spacing does not count: a code read box by box ("4 1 L 4 5") is the same text as "41L45".

def text_missing(reference, reading, n=5):
    """Share of the reading absent from the reference: runs of n letters and digits, each counted as often as it occurs.
    It includes the reading's own misreads."""
    g, h = alnum(reference), alnum(reading)
    have = collections.Counter(g[i:i + n] for i in range(len(g) - n + 1)); gone = 0
    for i in range(len(h) - n + 1):
        if have[h[i:i + n]]:
            have[h[i:i + n]] -= 1
        else:
            gone += 1
    return gone / max(len(h) - n + 1, 1)


def in_order(reference, reading):
    """Share of the reading's letters and digits that the reference holds in the same order."""
    g, h = alnum(reference), alnum(reading)
    return sum(b.size for b in difflib.SequenceMatcher(None, g, h, autojunk=False).get_matching_blocks()) / max(len(h), 1)


FORM_MARKS = {"box": r"[☐☒☑□■]", "blank": r"(?:\.\s?){5,}|_{5,}|…{3,}"}


def form_gap(reference, reading):
    """Tick boxes and blanks (rows of dots or underscores) the reading holds beyond the reference, by kind."""
    return {k: len(re.findall(r, reading)) - len(re.findall(r, reference)) for k, r in FORM_MARKS.items()}


def is_skipped(text, reference):
    """The output holds less than 90% of the reference characters. Meaningful on upright pages only."""
    return len(squash(text)) < SKIP_RATIO * len(squash(reference))


def cer(reference, text):
    """Character error rate on the alnum stream, capped at 1."""
    import jiwer
    r = alnum(reference)
    return min(1.0, jiwer.cer(r, alnum(text))) if r else 0.0


def recall(counts, ids):
    """counts: page id -> (entities found, entities total). Entities are pooled over the pages in ids."""
    return 100 * sum(counts[i][0] for i in ids) / max(sum(counts[i][1] for i in ids), 1)


def bootstrap(stat, ids, n=BOOT_N, seed=0):
    """95% percentile interval of stat(page ids resampled with replacement).
    Pages are the unit: a failure takes out a page or a block, not a single entity."""
    rng = random.Random(seed)
    xs = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]
