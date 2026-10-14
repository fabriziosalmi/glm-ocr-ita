"""python experiments/omnidoc_compare.py TAG_A TAG_B : the paired difference A - B of OmniDocBench edit distance, times 100 (positive =
B better), on the English and the Chinese pages apart, from work/omnidoc_eval/TAG.json (experiments/omnidoc_eval.py). Pages are those
with matched blocks on both sides; bootstrap over pages (2000, seed 0) as everywhere else; verdict by rule 4 (score.verdict, margin -1.5)."""
import json, statistics, sys
from glmocr import corpus, metrics, score
from glmocr.corpus import WORK

a, b = sys.argv[1:3]
meta = {m["id"]: m["lang"] for m in corpus.load_meta("external/omnidoc_200")}   # the evaluator keys pages by image file name
ea, eb = (json.load(open(WORK / "omnidoc_eval" / f"{t}.json", encoding="utf-8")) for t in (a, b))
print(f"A - B (x100, positive = B better), A = {a}, B = {b}")
for metric in ("text_block_per_page", "reading_order_per_page"):
    for lang in ("en", "zh"):
        ids = [i for i in ea[metric] if i in eb[metric] and meta.get(i.rsplit(".", 1)[0]) == lang]
        d = lambda s: 100 * statistics.mean(ea[metric][i] - eb[metric][i] for i in s)
        lo, hi = metrics.bootstrap(d, ids)
        mean = lambda e: 100 * statistics.mean(e[metric][i] for i in ids)
        print(f"{metric[:-9]:14} {lang}  {len(ids):3} pages  A {mean(ea):5.1f}  B {mean(eb):5.1f}  diff {d(ids):+.1f} [{lo:+.1f}, {hi:+.1f}]  {score.verdict((lo, hi))}")
