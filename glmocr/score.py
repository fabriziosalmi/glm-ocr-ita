"""Score saved runs against a page set. Returns plain dicts; cli.py prints them and tests/ check them against the published tables."""
import collections
import statistics

from . import corpus, metrics


def _counts(meta, pages):
    return {m["id"]: (metrics.found(m["entities"], pages[m["id"]]["text"]), len(m["entities"])) for m in meta if m["id"] in pages}


def summarize(corpus_name, tag, upright=False, with_cer=True):
    """Entity recall with its interval, by level and by entity type, the two defect counts, speed and memory."""
    meta = corpus.load_meta(corpus_name, upright)
    pages = corpus.load_run(tag)["pages"]
    counts = _counts(meta, pages)
    ids = list(counts)
    meta = [m for m in meta if m["id"] in counts]
    lo, hi = metrics.bootstrap(lambda s: metrics.recall(counts, s), ids)
    up = [m["id"] for m in meta if not m.get("rot90")]
    levels = {}
    for level in sorted({m["level"] for m in meta}):
        li = [m["id"] for m in meta if m["level"] == level]
        levels[level] = {"recall": metrics.recall(counts, li), "pages": len(li),
                         "cer": 100 * statistics.mean(metrics.cer(corpus.reference(corpus_name, i), pages[i]["text"]) for i in li) if with_cer else None}
    types = collections.defaultdict(lambda: [0, 0])
    for m in meta:
        h = metrics.squash(pages[m["id"]]["text"])
        for e in m["entities"]:
            types[e.get("type") or "untyped"][0] += metrics.squash(e["text"]) in h
            types[e.get("type") or "untyped"][1] += 1
    flips, amounts, acc_found, acc_total = 0, 0, 0, 0
    for m in meta:
        f, n = metrics.notation_flips(m["entities"], pages[m["id"]]["text"])
        flips, amounts = flips + f, amounts + n
        f, n = metrics.accent_recall(corpus.reference(corpus_name, m["id"]), pages[m["id"]]["text"])
        acc_found, acc_total = acc_found + f, acc_total + n
    langs = {l: metrics.recall(counts, [m["id"] for m in meta if m["lang"] == l]) for l in sorted({m["lang"] for m in meta})}
    skipped = [i for i in up if metrics.is_skipped(pages[i]["text"], corpus.reference(corpus_name, i))]
    last_missing = [i for i in ids if metrics.last_line_missing(corpus.reference(corpus_name, i), pages[i]["text"])]
    # a loop the reader cut as it started is still a loop: the page was not read to the end
    loops = [i for i in ids if pages[i].get("loop_cut") is not None or metrics.is_loop(pages[i]["text"], pages[i]["gen_tokens"])]
    return {"tag": tag, "pages": len(ids), "entities": sum(c[1] for c in counts.values()),
            "recall": metrics.recall(counts, ids), "ci": (lo, hi), "recall_upright": metrics.recall(counts, up),
            "levels": levels, "types": {t: 100 * f / max(n, 1) for t, (f, n) in sorted(types.items())},
            "skipped": skipped, "last_line_missing": last_missing, "upright": len(up), "loops": loops, "langs": langs,
            "notation_flips": flips, "notation_candidates": amounts, "accent_recall": 100 * acc_found / max(acc_total, 1), "accent_words": acc_total,
            "s_page": statistics.mean(pages[i]["s"] for i in ids),
            # runs read by regions (experiments/region_format.py) keep neither: nan, not a number made up
            "gen_tps": statistics.mean(pages[i].get("gen_tps", float("nan")) for i in ids),
            "peak_gb": max((pages[i]["peak_gb"] for i in ids if "peak_gb" in pages[i]), default=float("nan"))}


def compare(corpus_name, tag_a, tag_b, upright=False):
    """Paired difference B - A in entity recall on the pages both runs have, with the pages that lost or gained entities."""
    meta = corpus.load_meta(corpus_name, upright)
    a, b = _counts(meta, corpus.load_run(tag_a)["pages"]), _counts(meta, corpus.load_run(tag_b)["pages"])
    ids = [i for i in a if i in b]
    diff = lambda s: metrics.recall(b, s) - metrics.recall(a, s)
    lo, hi = metrics.bootstrap(diff, ids)
    lost = sorted((b[i][0] - a[i][0], i) for i in ids if b[i][0] < a[i][0])
    return {"a": tag_a, "b": tag_b, "pages": len(ids), "recall_a": metrics.recall(a, ids), "recall_b": metrics.recall(b, ids),
            "diff": diff(ids), "ci": (lo, hi), "lost": lost, "gained": sum(b[i][0] > a[i][0] for i in ids)}


def verdict(ci, margin=-1.5):
    """Rule 4 of docs/ROADMAP.md, on the interval of B - A."""
    lo, hi = ci
    return "better" if lo > 0 else "worse" if hi < 0 else "not worse" if lo > margin else "not shown"


def adaptive(corpus_name, tags, upright=False):
    """What would re-reading a page at another scale buy? tags[0] is the base run, the others are the same pages at other scales.
      longest  read at every scale, keep the longest output that is not a loop (a skipped block makes the text shorter)
      oracle   keep the output with most entities found: needs the ground truth, so it is only the ceiling
      retry    base; read the others too only when base looped, then keep the longest sane output"""
    runs = {t: corpus.load_run(t)["pages"] for t in tags}
    meta = [m for m in corpus.load_meta(corpus_name, upright) if all(m["id"] in runs[t] for t in tags)]
    total = sum(len(m["entities"]) for m in meta)
    base = tags[0]
    looped = lambda m, t: metrics.is_loop(runs[t][m["id"]]["text"], runs[t][m["id"]]["gen_tokens"])
    hits = lambda m, t: metrics.found(m["entities"], runs[t][m["id"]]["text"])
    size = lambda m, t: -1 if looped(m, t) else len(metrics.squash(runs[t][m["id"]]["text"]))
    secs = lambda m, t: runs[t][m["id"]]["s"]
    every = lambda m: sum(secs(m, t) for t in tags)
    row = lambda name, pick, cost: {"policy": name, "recall": 100 * sum(hits(m, pick(m)) for m in meta) / max(total, 1),
                                    "s_page": sum(cost(m) for m in meta) / len(meta)}
    rows = [row(f"only {t}", lambda m, t=t: t, lambda m, t=t: secs(m, t)) for t in tags]
    rows.append(row("longest", lambda m: max(tags, key=lambda t: size(m, t)), every))
    rows.append(row("oracle (needs ground truth)", lambda m: max(tags, key=lambda t: hits(m, t)), every))
    rows.append(row("retry when base looped", lambda m: max(tags, key=lambda t: size(m, t)) if looped(m, base) else base,
                    lambda m: every(m) if looped(m, base) else secs(m, base)))
    return {"pages": len(meta), "entities": total, "base_loops": sum(looped(m, base) for m in meta), "rows": rows}
