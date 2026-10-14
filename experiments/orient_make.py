"""python experiments/orient_make.py [hard|held] : orientation test set in work/orient_<set>/. Undo the 90deg pages exactly, then save every
page rotated by 0/90/180/270 (PIL, counter-clockwise). Label = counter-clockwise degrees applied to the upright page."""
import json, os, pathlib, sys
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from PIL import Image
SRC = sys.argv[1] if len(sys.argv) > 1 else "hard"; OUT = pathlib.Path(f"work/orient_{SRC}"); OUT.mkdir(parents=True, exist_ok=True); rows = []
for m in json.load(open(f"{SRC}/meta.json")):
    im = Image.open(f"{SRC}/{m['id']}.png").convert("RGB")
    if m["rot90"]: im = im.rotate(-90, expand=True)   # exact inverse of rotate(90, expand=True)
    for k in (0, 90, 180, 270):
        p = OUT / f"{m['id']}_r{k}.png"; (im if k == 0 else im.rotate(k, expand=True)).save(p)
        rows.append({"file": str(p), "id": m["id"], "level": m["level"], "label": k})
json.dump(rows, open(OUT / "labels.json", "w")); print(len(rows), "images in", OUT)
