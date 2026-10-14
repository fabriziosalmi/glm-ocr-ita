"""python generators/make_sroie.py [test|train] : put the SROIE 2019 receipts (external/urbikn_sroie-v2, see docs/EXTERNAL.md) in the format of this
repository -> external/sroie_<split>/{id}.png, {id}.txt, meta.json. Nothing is written outside external/, which is not in git:
the receipts are not ours to redistribute. Ground truth text = the line transcriptions of the dataset, in file order.
Entities = the four key fields of the dataset (company, date, address, total); empty fields are dropped.
Half of the scans are larger than our pages and some are 16 times the area: an image whose long side exceeds 1754 px (A4 at 150 dpi,
the size of hard/ and held/) is reduced to 1754. Smaller images are left as they are. Fixed before the first read of the set."""
import json, os, pathlib, sys
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from PIL import Image
split = sys.argv[1] if len(sys.argv) > 1 else "test"
SRC = pathlib.Path("external/urbikn_sroie-v2/data/SROIE2019") / split; OUT = pathlib.Path(f"external/sroie_{split}"); OUT.mkdir(exist_ok=True)
meta = []
for img in sorted((SRC / "img").glob("*.jpg")):
    i = img.stem; ent = json.loads((SRC / "entities" / f"{i}.txt").read_text(encoding="utf-8", errors="replace"))
    lines = [ln.split(",", 8)[8] for ln in (SRC / "box" / f"{i}.txt").read_text(encoding="utf-8", errors="replace").splitlines() if ln.count(",") >= 8]
    im = Image.open(img).convert("RGB"); orig = list(im.size); k = 1754 / max(im.size)
    if k < 1: im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(OUT / f"{i}.png"); (OUT / f"{i}.txt").write_text("\n".join(lines), encoding="utf-8")
    meta.append({"id": i, "gt": "dataset", "level": 1, "tpl": "receipt", "lang": "en", "rot90": False, "size": list(im.size), "orig_size": orig,
                 "entities": [{"type": k, "text": v.strip()} for k, v in ent.items() if isinstance(v, str) and v.strip()]})
json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
print(len(meta), "receipts;", sum(len(m["entities"]) for m in meta), "entities ->", OUT)
