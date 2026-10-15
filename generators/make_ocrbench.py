"""python generators/make_ocrbench.py [--src ~/Documents/git/ocr-bench-v1] : the owner's synthetic stress set ocr-bench-v1 (100 pages,
10 kinds of document of 10 pages each, 45 Italian, 45 English, 10 mixed), cut in two before any model read a page of it.
-> external/ocrbench_test (50 pages) and external/ocrbench_train (50, kept for fine-tuning iterations), as the owner asked on 2026-10-15.

The cut: within each kind (the set's "category"), its pages sorted by language and id, shuffled with a fixed seed within each
language, then dealt to test and train in turn, starting with test in one kind and train in the next: each part has 5 pages of
every kind and the set's language mix.
The pages are its PNG renders (benchmarks/png, 1654 x 2338, 200 dpi). Its PDFs are not used: each is the same PNG in a PDF with no
text layer, so glmocr/pdf.py would hand every page to the model.
Entities are drawn from the ground truth as for itdoc_14_scientific (generators/split_chatgpt.py): every distinct word of 8 letters
or more, the Italian codes, dates and amounts, and every number written with a decimal separator.
Meta: lang it / en / mix, tpl = the set's category, level 1 for every page (the set's difficulty_score is kept as "difficulty")."""
import argparse, collections, json, os, pathlib, random, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent; os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "generators"))
from split_chatgpt import scientific_entities

SEED = 20261015
LANG = {"it": "it", "en": "en", "mixed": "mix"}
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="~/Documents/git/ocr-bench-v1")
src = pathlib.Path(ap.parse_args().src).expanduser() / "benchmarks" / "png"
docs = [json.load(open(d / "metadata.json", encoding="utf-8")) for d in sorted(src.glob("doc_*"))]
assert len(docs) == 100, len(docs)
rng = random.Random(SEED); parts = {"test": [], "train": []}
for c, cat in enumerate(sorted({d["category"] for d in docs})):
    pages = []
    for lang in ("it", "en", "mixed"):
        group = sorted((d for d in docs if d["category"] == cat and d["language"] == lang), key=lambda d: d["id"])
        rng.shuffle(group); pages += group
    for k, d in enumerate(pages):
        parts["test" if (k + c) % 2 == 0 else "train"].append(d)
for part, pages in parts.items():
    out = pathlib.Path("external") / f"ocrbench_{part}"; out.mkdir(exist_ok=True); meta = []
    for d in sorted(pages, key=lambda d: d["id"]):
        i = d["id"]; shutil.copy2(src / i / f"{i}.png", out / f"{i}.png"); shutil.copy2(src / i / f"{i}.txt", out / f"{i}.txt")
        text = (out / f"{i}.txt").read_text(encoding="utf-8")
        meta.append({"id": i, "gt": "ocr-bench-v1", "level": 1, "tpl": d["category"], "lang": LANG[d["language"]], "rot90": False,
                     "difficulty": d["difficulty_score"], "title": d["title"], "size": [d["dimensions"]["width_px"], d["dimensions"]["height_px"]],
                     "entities": scientific_entities(text)})
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out, len(meta), "pages,", sum(len(m["entities"]) for m in meta), "entities;",
          dict(collections.Counter(m["lang"] for m in meta)), "kinds", sorted(collections.Counter(m["tpl"] for m in meta).values()))
