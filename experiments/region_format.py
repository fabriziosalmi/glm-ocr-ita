"""work/layout_venv/bin/python experiments/region_format.py TAG : the regions read by experiments/region_read.py (results/TAG_regions.json)
put together into one markdown page each by GLM-OCR's own result formatter (SDK `glmocr` 0.1.5, Apache-2.0, default settings: titles
as headings, formulas between $$, formula numbers and hyphenated text blocks merged, bullet points) -> results/TAG.json, in the
format of `glmocr run`, so that the scorers read it as any other run. "gen_tokens" is the largest region's count (the loop check
compares it with one read's budget); "gen_tokens_total" is the page's. Image regions are left out of the text.
Runs in work/layout_venv, as experiments/layout_regions.py, and checks that it imported the SDK and not this repository's glmocr."""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != ROOT]
import glmocr
from glmocr.config import load_config
from glmocr.postprocess.result_formatter import ResultFormatter

assert "site-packages" in glmocr.__file__, f"imported {glmocr.__file__}, not GLM-OCR's SDK"
tag = sys.argv[1]
run = json.load(open(ROOT / "results" / f"{tag}_regions.json", encoding="utf-8"))
fmt = ResultFormatter(load_config().pipeline.result_formatter)
pages = {}
for i, p in run["pages"].items():
    regions = [{"index": r["index"], "label": r["label"], "content": r.get("content"), "bbox_2d": r["bbox_2d"], "task_type": r["task_type"]}
               for r in p["regions"] if r["task_type"] != "skip" and r.get("content") is not None]
    _, md, _ = fmt.process([regions])
    read = [r for r in p["regions"] if "content" in r]
    pages[i] = {"text": md, "s": p["s"] + p["layout_s"], "regions": len(read), "gen_tokens": max((r["gen_tokens"] for r in read), default=0), "gen_tokens_total": sum(r["gen_tokens"] for r in read),
                "looped": any(r["looped"] for r in read), "loop_cut": None}
json.dump({**{k: v for k, v in run.items() if k != "pages"}, "reader": "regions (GLM-OCR SDK layout and formatter)", "pages": pages},
          open(ROOT / "results" / f"{tag}.json", "w"), ensure_ascii=False, indent=1)
print(len(pages), "pages ->", ROOT / "results" / f"{tag}.json")
