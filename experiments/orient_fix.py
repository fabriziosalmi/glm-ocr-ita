"""python experiments/orient_fix.py [held] : straighten the rotated pages of a set with the classifier's own answer (work/orient_<set>/pred.json,
from orient_eval.py) -> work/<set>_fix/, a small set holding only those pages. OCR it with glmocr run --corpus work/<set>_fix, then
`python experiments/orient_fix.py <set> --merge BASE_TAG FIX_TAG` writes results/BASE_TAG_oriented.json: the base run with those pages replaced."""
import json, os, pathlib, shutil, sys
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from PIL import Image
SRC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "held"; OUT = pathlib.Path(f"work/{SRC}_fix")
meta = [m for m in json.load(open(f"{SRC}/meta.json")) if m["rot90"]]
if "--merge" in sys.argv:
    base, fix = sys.argv[sys.argv.index("--merge") + 1:][:2]; b = json.load(open(f"results/{base}.json")); f = json.load(open(f"results/{fix}.json"))["pages"]
    b["pages"].update(f); b["tag"] = base + "_oriented"; json.dump(b, open(f"results/{base}_oriented.json", "w"), ensure_ascii=False, indent=1)
    print("replaced", len(f), "pages ->", f"results/{base}_oriented.json"); raise SystemExit
pred = {r["id"]: r for r in json.load(open(f"work/orient_{SRC}/pred.json")) if r["label"] == 90}   # label 90 = the page as it is in the set
OUT.mkdir(parents=True, exist_ok=True)
for m in meta:
    p = pred[m["id"]]["pred"]; im = Image.open(f"{SRC}/{m['id']}.png").convert("RGB")
    (im.rotate(p, expand=True) if p else im).save(OUT / f"{m['id']}.png"); shutil.copy(f"{SRC}/{m['id']}.txt", OUT)
    print(m["id"], "classifier says", p, "-> straightened" if p == 270 else "-> WRONG, page stays unreadable")
json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1); print(len(meta), "pages in", OUT)
