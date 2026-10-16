"""work/layout_venv/bin/python experiments/region_format.py TAG : the regions read by experiments/region_read.py (results/TAG_regions.json)
put together into one markdown page each by GLM-OCR's own result formatter (SDK `glmocr` 0.1.5, Apache-2.0, default settings: titles
as headings, formulas between $$, formula numbers and hyphenated text blocks merged, bullet points) -> results/TAG.json, in the
format of `glmocr run`, so that the scorers read it as any other run. "gen_tokens" is the largest region's count (the loop check
compares it with one read's budget); "gen_tokens_total" is the page's. Image regions are left out of the text.
Runs in work/layout_venv, as experiments/layout_regions.py; the formatting is glmocr/layout_worker.py's, as in `glmocr infer --regions`."""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("layout_worker", ROOT / "glmocr" / "layout_worker.py")
lw = importlib.util.module_from_spec(spec); spec.loader.exec_module(lw)
lw.sdk()
tag = sys.argv[1]
run = json.load(open(ROOT / "results" / f"{tag}_regions.json", encoding="utf-8"))
fmt = lw.formatter()
pages = {}
for i, p in run["pages"].items():
    md = lw.format_page(fmt, p["regions"])
    read = [r for r in p["regions"] if "content" in r]
    pages[i] = {"text": md, "s": p["s"] + p["layout_s"], "regions": len(read), "gen_tokens": max((r["gen_tokens"] for r in read), default=0), "gen_tokens_total": sum(r["gen_tokens"] for r in read),
                "looped": any(r["looped"] for r in read), "loop_cut": None}
json.dump({**{k: v for k, v in run.items() if k != "pages"}, "reader": "regions (GLM-OCR SDK layout and formatter)", "pages": pages},
          open(ROOT / "results" / f"{tag}.json", "w"), ensure_ascii=False, indent=1)
print(len(pages), "pages ->", ROOT / "results" / f"{tag}.json")
