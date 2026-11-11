"""python generators/make_mdpbench.py : the Italian pages of MDPBench's public set (Delores-Lin/MDPBench on Hugging Face, revision
252df132, Apache-2.0; docs/EXTERNAL.md) in the format of this repository -> external/mdp_it_dev/ and external/mdp_it_test/:
{id}.png, {id}.txt, meta.json, plus gt.json, the official annotation of those pages for the benchmark's own evaluator.

What the set adds: real phone photographs of Italian documents (indoor and outdoor light, creases, bends, shadows, flash), which no
other set of this repository has. 40 documents (papers, books, exam papers, financial reports, magazines, newspapers, notes, slides,
textbooks), each as one digital image and three photographs: 160 images. Ground truth by the benchmark's authors: "expert model
labeling, manual correction, and human verification".

Split by document, fixed before any model read a page (2026-11-11): the documents of each type are shuffled with seed 20261111 and
dealt in turn to dev and test, carrying on the turn from one type to the next, so that both halves hold 20 documents and every
image of a document is on one side only. Images are kept at their own size (photographs 2000 to 2500 px on the long side, digital
pages up to 4000), converted to PNG; a photograph is level 2, a digital page level 1.
Ground truth (<id>.txt): titles, text blocks, headers, footers, page numbers and captions, in the annotation's reading order; tables
and formulas are left out of it and of the entities. Entities as for DocAtlas (generators/make_docatlas.py, entities()): every
distinct word of 8 letters or more, plus dates, act numbers, amounts, VAT numbers, tax codes, IBAN, CIG and e-mail addresses."""
import collections, json, os, pathlib, random, re, sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent; os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "generators"))
from make_docatlas import entities  # noqa: E402

SRC = pathlib.Path("external/mdpbench_src"); SEED = 20261111
TEXT = ("title", "text_block", "header", "footer", "page_number", "figure_caption", "table_caption", "table_footnote", "figure_footnote")
gt = [p for p in json.load(open(SRC / "MDPBench_public.json", encoding="utf-8")) if p["page_info"]["image_path"].startswith("it_")]
doc = lambda p: re.match(r"(it_[a-z]+_\d+)", p["page_info"]["image_path"]).group(1)
kind = lambda d: d.split("_")[1]
docs = sorted({doc(p) for p in gt})
rng = random.Random(SEED); side = {}; turn = 0
for k in sorted({kind(d) for d in docs}):
    ds = sorted(d for d in docs if kind(d) == k); rng.shuffle(ds)
    for d in ds:
        side[d] = ("dev", "test")[turn % 2]; turn += 1
for half in ("dev", "test"):
    out = pathlib.Path(f"external/mdp_it_{half}"); out.mkdir(exist_ok=True)
    pages = sorted((p for p in gt if side[doc(p)] == half), key=lambda p: p["page_info"]["image_path"])
    meta = []
    for p in pages:
        name = p["page_info"]["image_path"]; i = pathlib.Path(name).stem
        photo = i != doc(p)
        img = Image.open(SRC / "MDPBench_img_public" / name).convert("RGB"); img.save(out / f"{i}.png")
        dets = sorted(p["layout_dets"], key=lambda d: int(d["order"]) if str(d.get("order", "")).lstrip("-").isdigit() else 10 ** 6)
        text = "\n".join(d["text"].strip() for d in dets if d.get("category_type") in TEXT and d.get("text", "").strip())
        (out / f"{i}.txt").write_text(text, encoding="utf-8")
        meta.append({"id": i, "gt": "mdpbench", "level": 2 if photo else 1, "tpl": kind(doc(p)), "lang": "it", "rot90": False,
                     "document": doc(p), "capture": i[len(doc(p)) + 1:] if photo else "digital", "size": list(img.size),
                     "image": name, "entities": entities(text)})
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(pages, open(out / "gt.json", "w", encoding="utf-8"), ensure_ascii=False)
    n = collections.Counter(e["type"] for m in meta for e in m["entities"])
    print(out, len(meta), "images,", len({m["document"] for m in meta}), "documents,", sum(n.values()), "entities", dict(n),
          dict(collections.Counter(m["tpl"] for m in meta)))
