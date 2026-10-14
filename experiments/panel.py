"""python experiments/panel.py read MODEL NAME [--scale 1.0] [--adapter DIR] [--pages 5] : read the panel with one configuration
python experiments/panel.py score BASE NAME... [--pages 5]                                : each NAME against BASE, paired

The panel is a small fixed sample for following a fine-tuning step by step: the first pages of the sampling order
(glmocr/corpus.py) of every page set that is not a test set. Two groups, never pooled:
  outside  spark and chatgpt: renderers no training page comes from. The number that counts.
  inside   gemini: the renderer of the training pages, on pages held back from training (experiments/ft_data.py --hold).
  english  the English pages of spark/itdoc_a#60 that are not already in "outside". The first pages of a sampling order
           are mostly Italian (4 English pages of 25 outside, none of 30 inside), too few to see English break: this
           group is the guard for the stopping rule of docs/ROADMAP.md. Its score is taken on these pages plus the
           English pages of "outside".
--pages 5 is the quick panel for intermediate steps, --pages 10 the whole one; the 5 are the first 5 of the 10.
A panel is a signal, not a verdict: with 5 or 10 pages per set the intervals are several points wide.
Reads use the reader's default prompt, the official one since 2026-10-14: names ending in _op are official-prompt reads, and an
adapter trained with one prompt is compared with the baseline read with the same one.
Results: results/panel_<set>_<NAME>.json. Pages already read in the pilot (PILOT below) are taken from there, not read again."""
import argparse, json
from glmocr import corpus, metrics
OUTSIDE = ["spark/itdoc_a", "chatgpt/itdoc_a", "chatgpt/itdoc_b", "chatgpt/itdoc_d", "chatgpt/itdoc_e"]
GUARD = "spark/itdoc_a"
INSIDE = ["gemini/itdoc_a", "gemini/itdoc_b", "gemini/itdoc_c", "gemini/itdoc_d", "gemini/itdoc_e", "gemini/itdoc_f"]
PILOT = {("spark/itdoc_a", "bf16_1.0"): "sparkA_bf16_1.0", ("spark/itdoc_a", "v8d8e8_1.0"): "sparkA_v8d8e8_1.0", ("spark/itdoc_a", "v16d4e4_1.0"): "sparkA_v16d4e4_1.0",
         ("chatgpt/itdoc_a", "bf16_1.0"): "chatgptA_bf16_1.0", ("chatgpt/itdoc_a", "v8d8e8_1.0"): "chatgptA_v8d8e8_1.0", ("chatgpt/itdoc_a", "v16d4e4_1.0"): "chatgptA_v16d4e4_1.0"}
PILOT.update({("spark/itdoc_a", "bf16_1.0_op"): "sparkA_bf16_1.0_op", ("spark/itdoc_a", "v8d8e8_1.0_op"): "sparkA_v8d8e8_1.0_op",
              ("chatgpt/itdoc_a", "bf16_1.0_op"): "chatgptA_bf16_1.0_op", ("chatgpt/itdoc_a", "v8d8e8_1.0_op"): "chatgptA_v8d8e8_1.0_op"})
tag = lambda s, name: PILOT.get((s, name), f"panel_{s.replace('/', '_')}_{name}")
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["read", "score", "list"]); ap.add_argument("args", nargs="*")
ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--adapter", default=None); ap.add_argument("--pages", type=int, default=5)
a = ap.parse_args()
for s in OUTSIDE + INSIDE:
    assert corpus.role(s) in ("dev", "train"), f"{s} is not dev or train: a test set never enters the panel"

def members(n):
    """[(group, set, page meta)] in reading order."""
    out = [("outside", s, m) for s in OUTSIDE for m in corpus.load_meta(f"{s}#{n}")] + [("inside", s, m) for s in INSIDE for m in corpus.load_meta(f"{s}#{n}")]
    seen = {(s, m["id"]) for _, s, m in out}
    return out + [("english", GUARD, m) for m in corpus.load_meta(f"{GUARD}#60") if m["lang"] == "en" and (GUARD, m["id"]) not in seen]

M = members(a.pages)
if a.cmd == "list":
    for g in ("outside", "inside", "english"):
        ms = [(s, m) for gg, s, m in M if gg == g]
        print(f"{g}: {len(ms)} pages, {sum(len(m['entities']) for _, m in ms)} entities, " + str({l: sum(m['lang'] == l for _, m in ms) for l in ('it', 'en')}))
    raise SystemExit

if a.cmd == "read":
    model, name = a.args
    todo = []
    for g, s, m in M:
        f = corpus.result_path(tag(s, name))
        if m["id"] not in (json.load(open(f))["pages"] if f.exists() else {}): todo.append((s, m))
    print(f"{name}: {len(M) - len(todo)} of {len(M)} panel pages already read")
    if not todo: raise SystemExit
    from glmocr.engine import Reader, environment
    reader = Reader(model, adapter=a.adapter, wait=True)   # one load for the whole panel; a Metal GPU timeout ends the run, the same command resumes it
    for s, m in todo:
        f = corpus.result_path(tag(s, name))
        res = json.load(open(f)) if f.exists() else {"tag": tag(s, name), "model": model, "corpus": s, "scale": a.scale, "adapter": a.adapter, "env": environment(), "pages": {}}
        p = reader.read(corpus.corpus_dir(s) / f"{m['id']}.png", a.scale); p.pop("looped")
        res["pages"][m["id"]] = p
        json.dump(res, open(f, "w"), ensure_ascii=False, indent=1)
        print(f"{name} {s:16s} {m['id']:44s} {p['s']:5.1f}s", flush=True)
    raise SystemExit

def counts(name):
    """(group, set, page id) -> (found, total, looped, under 90% of the text, language), for the pages NAME has read."""
    out = {}; cache = {}
    for g, s, m in M:
        f = corpus.result_path(tag(s, name))
        pages = cache.setdefault(s, json.load(open(f))["pages"] if f.exists() else {})
        if m["id"] in pages:
            p = pages[m["id"]]
            out[(g, s, m["id"])] = (metrics.found(m["entities"], p["text"]), len(m["entities"]), p.get("loop_cut") is not None or metrics.is_loop(p["text"], p["gen_tokens"]),
                                    metrics.is_skipped(p["text"], corpus.reference(s, m["id"])), m["lang"])
    return out

base, names = a.args[0], a.args[1:]
B = counts(base)
rec = lambda c, ks: metrics.recall({k: c[k][:2] for k in ks}, ks)
GROUPS = {"outside": lambda k: k[0] == "outside", "inside": lambda k: k[0] == "inside", "english": lambda k: k[0] in ("outside", "english") and B[k][4] == "en"}
print(f"panel of {a.pages} pages per set; B - A with A = {base}. A signal, not a verdict.")
print(f"{'configuration':20s} {'group':8s} {'pages':>5s} {'entities':>8s} {'A':>6s} {'B':>6s} {'diff':>6s} {'95% CI':>15s} {'loops A/B':>9s} {'<90% A/B':>9s}")
for name in names:
    C = counts(name)
    for g, inside in GROUPS.items():
        ks = [k for k in B if k in C and inside(k)]
        if not ks: continue
        d = lambda sel: rec(C, sel) - rec(B, sel)
        lo, hi = metrics.bootstrap(d, ks)
        print(f"{name:20s} {g:8s} {len(ks):5d} {sum(B[k][1] for k in ks):8d} {rec(B, ks):5.1f}% {rec(C, ks):5.1f}% {d(ks):+5.1f}  [{lo:+5.1f}, {hi:+5.1f}] "
              f"{sum(B[k][2] for k in ks):4d}/{sum(C[k][2] for k in ks):<4d} {sum(B[k][3] for k in ks):4d}/{sum(C[k][3] for k in ks):<4d}")
        if g != "english":
            for s in sorted({k[1] for k in ks}):
                sk = [k for k in ks if k[1] == s]
                print(f"{'':20s}   {s:18s} {len(sk):3d} {'':8s} {rec(B, sk):5.1f}% {rec(C, sk):5.1f}% {d(sk):+5.1f}")
    # the stopping rule of docs/ROADMAP.md, on the pages both have read
    both = [k for k in B if k in C]
    en = [k for k in both if GROUPS["english"](k)]; out = [k for k in both if k[0] == "outside"]; wide = [k for k in both if k[0] in ("outside", "english")]
    it_out = [k for k in out if B[k][4] == "it"]
    why = []
    if en and rec(C, en) - rec(B, en) < -1.5: why.append(f"English {rec(C, en) - rec(B, en):+.1f} points on the guard")
    costly = [k for k in wide if C[k][2] and not B[k][2] and C[k][0] < B[k][0]]   # a new loop, cut or not, on a page that lost entities
    if costly: why.append(f"{len(costly)} new loop(s) that cost entities: " + ", ".join(k[2] for k in costly))
    harmless = sum(1 for k in wide if C[k][2] and not B[k][2]) - len(costly)
    if sum(C[k][3] for k in out) > sum(B[k][3] for k in out): why.append(f"pages under 90% of the text {sum(B[k][3] for k in out)} -> {sum(C[k][3] for k in out)}")
    missing = len([k for k in B if k not in C])
    print(f"{name:20s} stopping rule: " + ("STOP, " + "; ".join(why) if why else "passes")
          + (f" | Italian outside {rec(C, it_out) - rec(B, it_out):+.1f} ({'candidate' if rec(C, it_out) >= rec(B, it_out) else 'not a candidate'})" if it_out and not why else "")
          + (f" | new loops that cost no entity: {harmless}" if harmless else "")
          + (f" | INCOMPLETE: {missing} panel pages not read" if missing else ""))
