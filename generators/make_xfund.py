"""python generators/make_xfund.py [val|train] : put an Italian split of XFUND (val: 50 real forms, a test set; train: 149, a dev set;
docs/EXTERNAL.md) in the format of this repository -> external/xfund_it_<split>/{id}.png, {id}.txt, meta.json. Nothing is written outside external/, which is not in git:
the forms are CC BY-NC-SA 4.0 and are used for evaluation only.
Source: external/xfund_src/it.<split>.json and the images of it.<split>.zip (github.com/doc-analysis/XFUND, release v1.0).
Ground truth text = the text of every annotated segment, in reading order (top to bottom, then left to right, by the top of its box).
Entities = segments of at least 2 characters once whitespace is removed, typed by the dataset's own label: "answer" (the values
filled in a form, the number that matters), "question" (the printed field names), "header", "other". Short segments are left out
because a 1-character string matches anywhere.
Images: the long side is reduced to 1754 px (A4 at 150 dpi), as for SROIE; every XFUND form is 2480 x 3508. Fixed before the first read."""
import json, os, pathlib, sys
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from glmocr.metrics import squash
from PIL import Image

split = sys.argv[1] if len(sys.argv) > 1 else "val"
SRC = pathlib.Path("external/xfund_src"); OUT = pathlib.Path(f"external/xfund_it_{split}"); OUT.mkdir(exist_ok=True)
data = json.load(open(SRC / f"it.{split}.json", encoding="utf-8"))
meta = []
for doc in data["documents"]:
    i = pathlib.Path(doc["img"]["fname"]).stem
    segs = sorted((s for s in doc["document"] if s["text"].strip()), key=lambda s: (s["box"][1], s["box"][0]))
    im = Image.open(SRC / doc["img"]["fname"]).convert("RGB"); orig = list(im.size); k = 1754 / max(im.size)
    if k < 1: im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(OUT / f"{i}.png"); (OUT / f"{i}.txt").write_text("\n".join(s["text"].strip() for s in segs), encoding="utf-8")
    meta.append({"id": i, "gt": "dataset", "level": 1, "tpl": "form", "lang": "it", "rot90": False, "size": list(im.size), "orig_size": orig,
                 "entities": [{"type": s["label"], "text": s["text"].strip()} for s in segs if len(squash(s["text"])) >= 2]})
json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = lambda t: sum(e["type"] == t for m in meta for e in m["entities"])
print(len(meta), "forms;", sum(len(m["entities"]) for m in meta), "entities, of which answer", n("answer"), "question", n("question"),
      "header", n("header"), "other", n("other"), "->", OUT)
