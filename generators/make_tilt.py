"""python generators/make_tilt.py : pages of known tilt for glmocr/deskew.py -> external/tilt_c_rNN/ (NN = degrees), one folder per angle,
the same page ids in each so that runs at two angles pair up. The 12 L1 pages of spark/itdoc_c (dev, rendered without tilt) that follow
the 8 of spark/itdoc_c#40 in its sampling order, turned by NN degrees, counterclockwise on even positions and clockwise on odd ones,
canvas grown to keep every corner, white fill. Ground truth and entities as the source page. Fixed in docs/ROADMAP.md before any read."""
import json, os, pathlib, sys
from glmocr.corpus import ROOT, corpus_dir, load_meta, sampling_order; os.chdir(ROOT)
from PIL import Image

ANGLES = [0, 1, 2, 3, 6, 10, 15]
SRC = "spark/itdoc_c"
src = corpus_dir(SRC); first = {m["id"] for m in load_meta(f"{SRC}#40")}
pages = [m for m in sampling_order(load_meta(SRC)) if m["level"] == 1 and m["id"] not in first][:12]
assert len(pages) == 12 and not any(m["degradation"]["rotation_deg"] for m in pages)
for a in ANGLES:
    out = pathlib.Path(f"external/tilt_c_r{a:02d}"); out.mkdir(exist_ok=True); meta = []
    for i, m in enumerate(pages):
        deg = a if i % 2 == 0 else -a
        im = Image.open(src / f"{m['id']}.png").convert("RGB")
        if deg:
            im = im.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor="white")
        im.save(out / f"{m['id']}.png")
        (out / f"{m['id']}.txt").write_bytes((src / f"{m['id']}.txt").read_bytes())
        meta.append({**m, "tilt_deg": deg, "source": SRC})
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
    print(out, len(meta), sorted({x["lang"] for x in meta}), im.size)
