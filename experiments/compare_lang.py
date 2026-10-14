"""python experiments/compare_lang.py CORPUS TAG_A TAG_B [--by lang|type] : the paired difference B - A of `glmocr compare`,
on each language apart (default) or on each entity type apart (e.g. the "answer" fields of XFUND forms).
Same bootstrap (pages resampled, 2000 times, seed 0) on the pages, or on the entities of that type; verdict by rule 4."""
import argparse
from glmocr import corpus, metrics, score

ap = argparse.ArgumentParser(); ap.add_argument("corpus"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--by", choices=["lang", "type"], default="lang"); a = ap.parse_args()
meta = corpus.load_meta(a.corpus)
pa, pb = corpus.load_run(a.a)["pages"], corpus.load_run(a.b)["pages"]
print(f"B - A, A = {a.a}, B = {a.b}, corpus = {a.corpus}, by {a.by}")
if a.by == "lang":
    groups = {g: ([m for m in meta if m["lang"] == g], None) for g in sorted({m["lang"] for m in meta})}
else:
    groups = {g: (meta, g) for g in sorted({e.get("type") or "untyped" for m in meta for e in m["entities"]})}
for g, (ms, typ) in groups.items():
    ms = [dict(m, entities=[e for e in m["entities"] if typ is None or (e.get("type") or "untyped") == typ]) for m in ms if m["id"] in pa and m["id"] in pb]
    ms = [m for m in ms if m["entities"]]
    ca, cb = score._counts(ms, pa), score._counts(ms, pb)
    ids = [m["id"] for m in ms]
    diff = lambda s: metrics.recall(cb, s) - metrics.recall(ca, s)
    ci = metrics.bootstrap(diff, ids)
    print(f"{g:9s} {len(ids):4d} pages {sum(c[1] for c in ca.values()):5d} entities  A {metrics.recall(ca, ids):5.1f}%  B {metrics.recall(cb, ids):5.1f}%  "
          f"diff {diff(ids):+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]  {score.verdict(ci)}")
