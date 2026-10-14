"""python experiments/crops.py BASE_TAG --corpus NAME [--model work/glm-ocr-bf16] : can cropping recover what a whole-page read loses?
No training. For the pages BASE_TAG has already read whole, read again:
  bands   the top 16% and the bottom 14% of the page as two separate small images (letterhead and footer)
  halves  the page as two overlapping halves (0-56% and 44-100%), each enlarged 1.4x (small print gets bigger)
and report entity recall by zone for: whole page, whole + bands, halves, halves + bands. Raw texts -> results/<BASE_TAG>_crops.json

Caveat: a policy that joins several readings holds more text, so a short entity has more chances to match by accident.
The scan stays whole-page-relative: zones come from the line an entity sits on in the ground truth (first 3 lines, last 3, rest)."""
import argparse
import collections
import json

from PIL import Image

from glmocr import corpus, metrics
from glmocr.engine import Reader

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--corpus", required=True)
ap.add_argument("--model", default=str(corpus.SOURCE))
ap.add_argument("--mem-gb", type=float, default=None)
ap.add_argument("--score-only", action="store_true", help="no model: score the crops already saved")
a = ap.parse_args()
d = corpus.corpus_dir(a.corpus)
meta = {m["id"]: m for m in corpus.load_meta(a.corpus)}
whole = corpus.load_run(a.base)["pages"]
out = corpus.result_path(f"{a.base}_crops")
res = json.load(open(out)) if out.exists() else {}
todo = [] if a.score_only else [i for i in whole if i not in res]
if todo:
    reader = Reader(a.model, mem_gb=a.mem_gb)
    def read(im):
        r = reader.read(im)
        return {"text": r["text"], "s": r["s"], "ptok": r["prompt_tokens"]}
    big = lambda c: c.resize((round(c.width * 1.4), round(c.height * 1.4)), Image.LANCZOS)
    for i in todo:
        im = Image.open(d / f"{i}.png").convert("RGB")
        w, h = im.size
        res[i] = {"top": read(im.crop((0, 0, w, round(.16 * h)))), "bottom": read(im.crop((0, round(.86 * h), w, h))),
                  "half1": read(big(im.crop((0, 0, w, round(.56 * h))))), "half2": read(big(im.crop((0, round(.44 * h), w, h))))}
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        print(i, {k: round(v["s"], 1) for k, v in res[i].items()}, flush=True)

parts = {"whole page": [], "whole + bands": ["top", None, "bottom"], "two halves": ["half1", "half2"], "halves + bands": ["top", "half1", "half2", "bottom"]}
parts["whole page"] = [None]   # None = the whole-page read of BASE_TAG
text = lambda i, k: whole[i]["text"] if k is None else res[i][k]["text"]
secs = lambda i, k: whole[i]["s"] if k is None else res[i][k]["s"]
print(f"\n{'policy':16s} {'all':>12s} {'letterhead':>12s} {'body':>12s} {'footer':>12s} {'s/page':>7s}")
for name, keys in parts.items():
    z = collections.defaultdict(lambda: [0, 0])
    for i in res:
        h = metrics.squash("\n".join(text(i, k) for k in keys))
        lines = corpus.reference(a.corpus, i).split("\n")
        n = len(lines)
        for e in meta[i]["entities"]:
            li = next((k for k, l in enumerate(lines) if e["text"] in l), n // 2)
            zone = "letterhead" if li < 3 else "footer" if li >= n - 3 else "body"
            for k in (zone, "all"):
                z[k][0] += metrics.squash(e["text"]) in h
                z[k][1] += 1
    print(f"{name:16s} " + " ".join(f"{z[k][0]:4d}/{z[k][1]:<4d}{100*z[k][0]/max(z[k][1],1):3.0f}%".rjust(12) for k in ("all", "letterhead", "body", "footer"))
          + f" {sum(secs(i, k) for i in res for k in keys)/max(len(res), 1):7.1f}")
