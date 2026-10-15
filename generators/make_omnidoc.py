"""python generators/make_omnidoc.py : a fixed subset of OmniDocBench v1.5 (opendatalab/OmniDocBench, revision aa1ee96, Apache-2.0;
docs/EXTERNAL.md) in the format of this repository -> external/omnidoc_200/{id}.png, {id}.txt, meta.json, plus gt.json, the
official ground truth restricted to those pages, for the official evaluator (pdf_validation.py, end-to-end, text and reading order).
Pages: 150 in English and 50 in simplified Chinese, drawn with seed 20261014 in proportion to the data sources of each language
(largest remainder), sorted by image name before the draw. Fixed before any model read a page of the benchmark.
Images are copied as they are. meta.json carries no entities: the measure is the official one (normalized edit distance),
computed by the benchmark's own code on the markdown the model writes; `glmocr run` is used only to read and save the pages.
<id>.txt is the text of the page's text blocks in the order of the annotation, for reference only.

--out, --seed and --exclude draw further subsets the same way, from the pages not in the excluded sets (2026-10-15, before any read
of them): external/omnidoc_dev (seed 20261015, excluding omnidoc_200) for choosing a reading pipeline, and external/omnidoc_test2
(seed 20261016, excluding both) for the one test read that confirms it. Both in the benchmark's own proportions of language,
--want english=92,simplified_chinese=94,en_ch_mixed=14 (of 1651 pages: 755 English, 765 Chinese, 116 mixed, 15 other), since
Chinese is now in scope (decided by the owner on 2026-10-15); omnidoc_200 kept the earlier 150 English and 50 Chinese. omnidoc_200 has been read twice and decides nothing more.
With no option the script rebuilds omnidoc_200 as it was."""
import argparse
import collections, json, os, pathlib, random, shutil
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="external/omnidoc_200"); ap.add_argument("--seed", type=int, default=20261014)
ap.add_argument("--exclude", nargs="*", default=[], help="page sets built by this script whose pages are not drawn again")
ap.add_argument("--want", default="english=150,simplified_chinese=50", help="pages per language of the benchmark")
a = ap.parse_args(); WANT = {k: int(v) for k, v in (x.split("=") for x in a.want.split(","))}
SRC = pathlib.Path("external/omnidocbench_src"); OUT = pathlib.Path(a.out); OUT.mkdir(exist_ok=True)
taken = {m["image"] for d in a.exclude for m in json.load(open(pathlib.Path(d) / "meta.json", encoding="utf-8"))}
LANG = {"english": "en", "simplified_chinese": "zh", "en_ch_mixed": "mix"}
gt = json.load(open(SRC / "OmniDocBench.json", encoding="utf-8"))
rng = random.Random(a.seed)
chosen = []
for lang, n in WANT.items():
    pages = sorted((p for p in gt if p["page_info"]["page_attribute"]["language"] == lang and p["page_info"]["image_path"] not in taken),
                   key=lambda p: p["page_info"]["image_path"])
    by_src = collections.defaultdict(list)
    for p in pages:
        by_src[p["page_info"]["page_attribute"]["data_source"]].append(p)
    quota = {s: n * len(v) / len(pages) for s, v in by_src.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: (-(quota[s] - take[s]), s))[: n - sum(take.values())]:
        take[s] += 1
    for s in sorted(by_src):
        chosen += rng.sample(by_src[s], take[s])
meta = []
for p in chosen:
    name = p["page_info"]["image_path"]; i = pathlib.Path(name).stem; a = p["page_info"]["page_attribute"]
    shutil.copy(SRC / "images" / name, OUT / f"{i}.png") if name.endswith(".png") else __import__("PIL.Image").Image.open(SRC / "images" / name).convert("RGB").save(OUT / f"{i}.png")
    text = [d.get("text", "") for d in sorted(p["layout_dets"], key=lambda d: d.get("order") if d.get("order") is not None else 10 ** 6)
            if d.get("category_type") in ("text_block", "title") and d.get("text")]
    (OUT / f"{i}.txt").write_text("\n".join(text), encoding="utf-8")
    meta.append({"id": i, "gt": "omnidocbench", "level": 1, "tpl": a["data_source"], "lang": LANG[a["language"]], "rot90": False,
                 "size": [p["page_info"]["width"], p["page_info"]["height"]], "image": name, "entities": []})
json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(chosen, open(OUT / "gt.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(meta), "pages:", dict(collections.Counter((m["lang"], m["tpl"]) for m in meta)), "->", OUT)
