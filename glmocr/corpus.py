"""Where things are: page sets, saved runs, and the role each page set plays (datasets.toml).

A page set is a folder with meta.json, <id>.png and <id>.txt. `--corpus NAME` is looked up in the repository first
(hard, held), then as a path, then under the datasets root ($GLMOCR_DATASETS, default ~/Documents/git/datasets).

NAME#N is the first N pages of the set in its sampling order: a fixed shuffle, balanced by level and language, so that
every prefix is a balanced sample and a larger N only adds pages to a smaller one. The order depends on the page ids
alone; it was fixed on 2026-10-13, before any model read a commissioned set.
"""
import collections
import itertools
import json
import os
import pathlib
import random
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# work/: the lock, the server log, scratch files. In a checkout, next to the code; in an installed package (pip, Docker)
# ROOT is site-packages, so a folder of the user's, or GLMOCR_WORK.
WORK = pathlib.Path(os.environ.get("GLMOCR_WORK") or (ROOT / "work" if (ROOT / "pyproject.toml").exists() else "~/.cache/glmocr")).expanduser()
MODELS = ROOT / "models"
SOURCE = WORK / "glm-ocr-bf16"
LEDGER = ROOT / "docs" / "LEDGER.md"
DATASETS = pathlib.Path(os.environ.get("GLMOCR_DATASETS", "~/Documents/git/datasets")).expanduser()


def registry():
    return tomllib.loads((ROOT / "datasets.toml").read_text(encoding="utf-8"))["dataset"]


class Missing(SystemExit):
    """The page set is not on this machine."""


def split_name(name):
    base, _, n = name.partition("#")
    return base, int(n) if n else None


def sampling_order(meta, seed=20261013):
    """Round-robin over (level, language) groups, each shuffled with a fixed seed after sorting by id."""
    groups = collections.defaultdict(list)
    for m in sorted(meta, key=lambda m: m["id"]):
        groups[(m["level"], m["lang"])].append(m)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    # languages are not equal in number: interleave so that a prefix keeps the set's own language mix
    by_level = collections.defaultdict(list)
    for (level, lang), pages in sorted(groups.items()):
        by_level[level].append(pages)
    levels = []
    for level in sorted(by_level):
        pools = by_level[level]
        total = sum(len(p) for p in pools)
        tagged = sorted(((i + 0.5) / len(p), k, i) for k, p in enumerate(pools) for i in range(len(p)))
        levels.append([pools[k][i] for _, k, i in tagged])
        assert len(levels[-1]) == total
    return [m for row in itertools.zip_longest(*levels) for m in row if m is not None]


def corpus_dir(name):
    name = split_name(name)[0]
    for d in (ROOT / name, pathlib.Path(name).expanduser(), DATASETS / name):
        if (d / "meta.json").exists():
            return d.resolve()
    raise Missing(f"page set '{name}' not found in {ROOT}, as a path, or under {DATASETS}")


def role(name):
    """train, dev, test, spent or None, from datasets.toml. Matched on the resolved folder, so a path cannot dodge it."""
    d = corpus_dir(name)
    for key, spec in registry().items():
        try:
            if corpus_dir(key) == d:
                return spec["role"]
        except Missing:
            pass
    return None


def load_meta(name, upright=False):
    meta = json.loads((corpus_dir(name) / "meta.json").read_text(encoding="utf-8"))
    n = split_name(name)[1]
    if n is not None:
        meta = sampling_order(meta)[:n]
    return [m for m in meta if not (upright and m.get("rot90"))]


def escaped_breaks(text):
    """A ground truth written with the two characters backslash-n where its line breaks should be: it has such
    escapes and not a single real line break. Seen in every file of gemini/itdoc_a and itdoc_b, which hold no other escape."""
    return "\\n" in text and "\n" not in text


def read_reference(path):
    """The ground truth of one page. Escaped line breaks are turned into real ones, and only in a file that has no
    real one: a file with real line breaks is returned exactly as it is."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return text.replace("\\n", "\n") if escaped_breaks(text) else text


def reference(name, page_id):
    return read_reference(corpus_dir(name) / f"{page_id}.txt")


def result_path(tag):
    return RESULTS / f"{tag}.json"


def load_run(tag):
    """The whole saved run: {"tag", "model", "pages": {id: {"text", "s", "gen_tokens", ...}}, ...}."""
    return json.loads(result_path(tag).read_text(encoding="utf-8"))


def ledger_has(tag):
    return LEDGER.exists() and f"`{tag}`" in LEDGER.read_text(encoding="utf-8")
