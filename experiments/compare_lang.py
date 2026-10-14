"""python experiments/compare_lang.py CORPUS TAG_A TAG_B : the paired difference B - A of `glmocr compare`, on each language apart.
Same bootstrap (pages resampled, 2000 times, seed 0) on the pages of one language; verdict by rule 4."""
import sys
from glmocr import corpus, metrics, score

name, ta, tb = sys.argv[1:4]
meta = corpus.load_meta(name)
a, b = score._counts(meta, corpus.load_run(ta)["pages"]), score._counts(meta, corpus.load_run(tb)["pages"])
print(f"B - A, A = {ta}, B = {tb}, corpus = {name}")
for lang in sorted({m["lang"] for m in meta}):
    ids = [m["id"] for m in meta if m["lang"] == lang and m["id"] in a and m["id"] in b]
    diff = lambda s: metrics.recall(b, s) - metrics.recall(a, s)
    ci = metrics.bootstrap(diff, ids)
    print(f"{lang}  {len(ids):4d} pages  A {metrics.recall(a, ids):5.1f}%  B {metrics.recall(b, ids):5.1f}%  "
          f"diff {diff(ids):+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]  {score.verdict(ci)}")
