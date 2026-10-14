"""python generators/split_chatgpt.py : the two commissioned sets of 2026-10-14 night, cut into a test part and a training part.
-> external/realistic_test (20 pages) and external/realistic_train (80), from chatgpt/itdoc_13_realistic;
   external/scientific_test (20) and external/scientific_train (80), from chatgpt/itdoc_14_scientific.

Why. The owner of the project asked on 2026-10-15 to measure on 20 pages of each and keep the other 80 for fine-tuning
iterations. Roles are given per folder (datasets.toml), so the two parts are two folders. The test part is the first 20 pages
of the set's sampling order (glmocr.corpus.sampling_order, balanced by level and fixed by the page ids alone): the same pages
`NAME#20` would give. Cut before any model read a page of either set.
Entities. itdoc_13_realistic has its own (meta.json). itdoc_14_scientific has none: they are drawn from the ground truth as for
DocAtlas (generators/make_docatlas.py: every distinct word of 8 letters or more, and the Italian codes, dates and amounts), plus
every number written with a decimal separator ("12,5", "0.031"), the values a scientific page is read for.
Images and ground truth are copied unchanged (1240 x 1754, already A4 at 150 dpi)."""
import json, os, pathlib, re, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent; os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "generators"))
from glmocr import corpus
from make_docatlas import entities as docatlas_entities

TEST_N = 20
SETS = {"realistic": "chatgpt/itdoc_13_realistic", "scientific": "chatgpt/itdoc_14_scientific"}


def scientific_entities(text):
    flat = re.sub(r"\s+", " ", text)
    return docatlas_entities(text) + [{"type": "decimal", "text": t} for t in dict.fromkeys(re.findall(r"(?<![\d.,])\d+[.,]\d+(?![\d.,]*\d)", flat))]


if __name__ == "__main__":
    for short, name in SETS.items():
        src = corpus.corpus_dir(name); order = corpus.sampling_order(corpus.load_meta(name))
        for part, pages in (("test", order[:TEST_N]), ("train", order[TEST_N:])):
            out = pathlib.Path("external") / f"{short}_{part}"; out.mkdir(exist_ok=True); meta = []
            for m in pages:
                shutil.copy2(src / f"{m['id']}.png", out / f"{m['id']}.png"); shutil.copy2(src / f"{m['id']}.txt", out / f"{m['id']}.txt")
                if "entities" not in m:
                    m = {**m, "entities": scientific_entities(corpus.reference(name, m["id"]))}
                meta.append({**m, "rot90": m.get("rot90", False)})
            json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            levels = sorted({m["level"] for m in meta})
            print(out, len(meta), "pages,", sum(len(m["entities"]) for m in meta), "entities, per level", [sum(m["level"] == l for m in meta) for l in levels])
