"""python experiments/region_read.py MODEL TAG --corpus NAME [--wait] : every region of every page read alone, with our MLX reader
-> results/TAG_regions.json; then experiments/region_format.py turns it into results/TAG.json, one markdown text per page.

The regions come from experiments/layout_regions.py (GLM-OCR's SDK, PP-DocLayoutV3), work/layout/<set>/<id>.json. Each region the
SDK reads is cropped from the page on its box (0-1000 grid, as the SDK's crop_image_region does for a box) and read with the
prompt of its task (text, table, formula; engine.TASK_PROMPTS), as the SDK's default settings do: repetition penalty 1.1 on every
read and 8192 tokens. Regions the SDK skips (figures, charts) or abandons (headers, footers, page numbers) are not read.
Refuses a test set unless TAG is in docs/LEDGER.md, as `glmocr run` does. Resumes a partial run."""
import argparse, json, pathlib, time
from PIL import Image
from glmocr import corpus
from glmocr.engine import Reader, environment

PENALTY, MAX_TOKENS = 1.1, 8192   # the SDK's page_loader defaults
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--corpus", required=True)
ap.add_argument("--wait", action="store_true"); ap.add_argument("--mem-gb", type=float, default=None)
a = ap.parse_args()
if corpus.role(a.corpus) == "test" and not corpus.ledger_has(a.tag):
    raise SystemExit(f"'{a.corpus}' is a test set: write the question and the tag `{a.tag}` in docs/LEDGER.md first.")
src = corpus.corpus_dir(a.corpus); meta = corpus.load_meta(a.corpus); lay = corpus.WORK / "layout" / src.name
missing = [m["id"] for m in meta if not (lay / f"{m['id']}.json").exists()]
if missing:
    raise SystemExit(f"{len(missing)} pages have no regions in {lay}: run experiments/layout_regions.py first")
Image.MAX_IMAGE_PIXELS = None
out = corpus.RESULTS / f"{a.tag}_regions.json"
run = json.load(open(out)) if out.exists() else {"tag": a.tag, "model": a.model, "corpus": a.corpus, "penalty": PENALTY,
                                                 "max_tokens": MAX_TOKENS, "env": environment(), "pages": {}}
reader = Reader(a.model, max_tokens=MAX_TOKENS, mem_gb=a.mem_gb, wait=a.wait)


def crop(img, box):
    W, H = img.size; x1, y1, x2, y2 = box
    return img.crop((int(x1 * W / 1000), int(y1 * H / 1000), int(x2 * W / 1000), int(y2 * H / 1000)))


for m in meta:
    if m["id"] in run["pages"]:
        continue
    L = json.load(open(lay / f"{m['id']}.json")); img = Image.open(src / f"{m['id']}.png").convert("RGB"); t = time.time(); regions = []
    for r in L["regions"]:
        r = dict(r)
        if r["task_type"] in ("text", "table", "formula"):
            c = crop(img, r["bbox_2d"])
            if c.width > 1 and c.height > 1:
                o = reader.read(c, penalty=PENALTY, task=r["task_type"])
                r.update(content=o["text"], gen_tokens=o["gen_tokens"], looped=o["looped"], s=round(o["s"], 2))
        regions.append(r)
    run["pages"][m["id"]] = {"regions": regions, "s": time.time() - t, "layout_s": L["s"]}
    json.dump(run, open(out, "w"), ensure_ascii=False)
    n = sum("content" in r for r in regions)
    print(a.tag, m["id"][:45], f"{n} regions read", f"{time.time() - t:.1f}s", flush=True)
print("->", out)
