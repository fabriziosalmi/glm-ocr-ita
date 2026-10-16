"""The half of the region mode that runs in GLM-OCR's own SDK (zai-org/GLM-OCR, package `glmocr` 0.1.5, Apache-2.0), in its own
Python environment: its layout detector (PP-DocLayoutV3, PaddlePaddle/PP-DocLayoutV3_safetensors, Apache-2.0) finds the regions of a
page and their reading order, and its result formatter puts the regions' readings together into one markdown page. The reading
in between is done by glmocr/regions.py with the MLX reader, in this repository's environment.

Not imported by this package: it is started by glmocr/regions.py as `LAYOUT_PYTHON layout_worker.py [--keep] [--device mps]`, with
LAYOUT_PYTHON an environment where `pip install "glmocr[layout]==0.1.5"` was run. The SDK's package is also called glmocr: this
file's own folder is taken off sys.path, and it checks that it imported the SDK.
Protocol: one JSON object per line. It answers {"ready": SDK version} when loaded; then {"regions": image path} ->
{"regions": [...], "size": [w, h], "s": seconds}, {"format": [regions with "content"]} -> {"text": markdown}, an error as
{"error": message}. Its stdout carries the protocol only: anything the SDK prints goes to stderr.
--keep: the labels the SDK abandons are read as text instead (headers, footers, page numbers, footnotes, side text, references;
header and footer images are skipped as images are). experiments/layout_regions.py and region_format.py use the same functions."""
import json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
LABEL_FIELDS = ("index", "label", "task_type", "score", "bbox_2d")


def sdk():
    """The SDK, never this repository's package of the same name."""
    sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() not in (HERE, HERE.parent)]
    import glmocr
    assert "site-packages" in glmocr.__file__, f"imported {glmocr.__file__}, not GLM-OCR's SDK"
    from importlib.metadata import version
    return version("glmocr")


def detector(keep=False, device="mps"):
    from glmocr.config import load_config
    from glmocr.layout.layout_detector import PPDocLayoutDetector
    cfg = load_config().pipeline.layout
    cfg.device = device
    if keep:
        kept = cfg.label_task_mapping.pop("abandon")
        cfg.label_task_mapping["skip"] += [l for l in kept if l.endswith("_image")]
        cfg.label_task_mapping["text"] += [l for l in kept if not l.endswith("_image")]
    det = PPDocLayoutDetector(cfg); det.start()
    return det, cfg


def regions(det, img):
    (found,), _ = det.process([img])
    return [{k: r[k] for k in LABEL_FIELDS} for r in found]


def formatter():
    from glmocr.config import load_config
    from glmocr.postprocess.result_formatter import ResultFormatter
    return ResultFormatter(load_config().pipeline.result_formatter)


def format_page(fmt, read):
    """read: the page's regions, in the detector's order, each with its reading in "content". Skipped and unread regions are left out."""
    page = [{k: r[k] for k in ("index", "label", "content", "bbox_2d", "task_type")}
            for r in read if r["task_type"] != "skip" and r.get("content") is not None]
    _, md, _ = fmt.process([page])
    return md


def serve(keep, device):
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1); os.dup2(2, 1)   # the protocol alone on stdout
    version = sdk()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    det, _ = detector(keep, device); fmt = formatter()
    print(json.dumps({"ready": version}), file=out)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if "regions" in req:
                t = time.time(); img = Image.open(req["regions"]).convert("RGB")
                ans = {"regions": regions(det, img), "size": list(img.size), "s": round(time.time() - t, 2)}
            else:
                ans = {"text": format_page(fmt, req["format"])}
        except Exception as e:   # one bad page must not take the job down: the caller decides
            ans = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(ans, ensure_ascii=False), file=out)
    det.stop()


if __name__ == "__main__":
    serve("--keep" in sys.argv, sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "mps")
