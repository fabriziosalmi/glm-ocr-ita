"""work/layout_venv/bin/python experiments/layout_regions.py --corpus NAME [--device mps] [--keep] : the regions of every page of a page set, found by
GLM-OCR's own SDK (zai-org/GLM-OCR, package `glmocr` 0.1.5, Apache-2.0) with its default layout settings -> work/layout/<set>/<id>.json.

Why the SDK itself. Its numbers on OmniDocBench come from this step: PP-DocLayoutV3 (PaddlePaddle/PP-DocLayoutV3_safetensors,
Apache-2.0) finds the regions of a page and their reading order, and each region is then read alone with a prompt for its kind
(text, table, formula). Running the SDK's own detector and post-processing, rather than a copy of them, keeps that step as theirs.
Only the regions are taken from it; the reading is done by experiments/region_read.py with our MLX reader.
Settings: the SDK's default config.yaml (threshold 0.3, its NMS and box merging, its label-to-task mapping: headers, footers, page
numbers and footnotes abandoned, figures and charts skipped). Each region: label, task, score, box on a 0-1000 grid, in the order
the SDK gives them, which is its reading order.
--keep (2026-10-16): the labels the SDK abandons are read as text instead (headers, footers, page numbers, footnotes, side text,
references; header and footer images are skipped as images are) -> work/layout/<set>_keep/. Italian documents carry content there
(a Gazzetta page's issue line and dates, a form's footer codes); OmniDocBench does not score it.

Runs in its own environment (PyTorch, the SDK): work/layout_venv, built with `uv pip install "glmocr[layout]==0.1.5"`. The SDK's
package is also called glmocr: the functions are glmocr/layout_worker.py's, loaded by file, which check that glmocr is the SDK."""
import argparse, importlib.util, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("layout_worker", ROOT / "glmocr" / "layout_worker.py")
lw = importlib.util.module_from_spec(spec); spec.loader.exec_module(lw)   # the functions of the region mode, by file: not as glmocr
version = lw.sdk()   # takes this repository off sys.path and checks that glmocr is the SDK
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # one OmniDocBench page is 145 megapixels

ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True); ap.add_argument("--device", default="mps")
ap.add_argument("--keep", action="store_true", help="read headers, footers, page numbers and footnotes as text")
a = ap.parse_args()
src = pathlib.Path(a.corpus) if pathlib.Path(a.corpus).is_absolute() else ROOT / a.corpus
meta = json.load(open(src / "meta.json", encoding="utf-8"))
out = ROOT / "work/layout" / (src.name + ("_keep" if a.keep else "")); out.mkdir(parents=True, exist_ok=True)
det, cfg = lw.detector(a.keep, a.device)
for m in meta:
    f = out / f"{m['id']}.json"
    if f.exists():
        continue
    t = time.time(); img = Image.open(src / f"{m['id']}.png").convert("RGB")
    regions = lw.regions(det, img)
    json.dump({"id": m["id"], "size": list(img.size), "sdk": f"glmocr {version}", "model": cfg.model_dir, "device": a.device,
               "s": round(time.time() - t, 2), "regions": regions}, open(f, "w"), ensure_ascii=False)
    print(m["id"][:50], len(regions), "regions", f"{time.time() - t:.1f}s", flush=True)
det.stop()
print("->", out)
