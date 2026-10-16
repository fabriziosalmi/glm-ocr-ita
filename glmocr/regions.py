"""A page read by regions, after GLM-OCR's own pipeline: the regions and their order come from its SDK's layout detector, each
region is read alone by the MLX reader with the prompt of its kind (text, table, formula; engine.TASK_PROMPTS), and the SDK's result
formatter puts the readings together. Measured on OmniDocBench (docs/LEDGER.md, omnidoc_test2): overall_edit 64.1 for the whole
page, 93.7 by regions, mostly tables, newspapers and Chinese pages. Not the default: on real Italian forms (xfund_it_train) it
finds fewer entities than the whole page, -5.7 [-9.1, -2.7], where the detector misses a form's body; on the Gazzetta it is even
(docs/ROADMAP.md, Phase 4).

The SDK needs PyTorch, so it runs in its own Python environment, as glmocr/layout_worker.py, one process for the whole job:
    python3.12 -m venv .layout && .layout/bin/pip install "glmocr[layout]==0.1.5"
and GLMOCR_LAYOUT_PYTHON=.layout/bin/python (or --layout-python). The settings are the SDK's defaults, as measured: repetition
penalty 1.1 on every region (so no loop retry), 8192 tokens per region; figures and charts are not read. Headers, footers, page
numbers and footnotes, which the SDK abandons, are read unless keep=False."""
import json, os, pathlib, subprocess, sys, tempfile, time

PENALTY, MAX_TOKENS = 1.1, 8192   # the SDK's page_loader defaults
WORKER = pathlib.Path(__file__).with_name("layout_worker.py")


class LayoutError(RuntimeError):
    pass


def layout_python(given=None):
    python = given or os.environ.get("GLMOCR_LAYOUT_PYTHON")
    if not python:
        raise SystemExit("the region mode needs GLM-OCR's SDK in its own environment:\n"
                         '  python3.12 -m venv .layout && .layout/bin/pip install "glmocr[layout]==0.1.5"\n'
                         "then GLMOCR_LAYOUT_PYTHON=.layout/bin/python or --layout-python .layout/bin/python")
    return python


class Layout:
    """The SDK's layout detector and result formatter, in a process of their own (glmocr/layout_worker.py)."""
    def __init__(self, python=None, keep=True, device="mps"):
        self.keep = keep
        self.proc = subprocess.Popen([layout_python(python), str(WORKER), "--device", device] + (["--keep"] if keep else []),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.sdk = self._answer()["ready"]

    def _answer(self):
        line = self.proc.stdout.readline()
        if not line:
            raise LayoutError(f"the layout process ended (exit code {self.proc.wait()}); its messages are above")
        ans = json.loads(line)
        if "error" in ans:
            raise LayoutError(ans["error"])
        return ans

    def _ask(self, req):
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n"); self.proc.stdin.flush()
        return self._answer()

    def regions(self, image_path):
        return self._ask({"regions": str(image_path)})

    def format(self, read):
        return self._ask({"format": read})["text"]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close(); self.proc.wait(30)


def crop(img, box):
    """A region's box on the SDK's 0-1000 grid, cut from the page as its crop_image_region does."""
    W, H = img.size; x1, y1, x2, y2 = box
    return img.crop((int(x1 * W / 1000), int(y1 * H / 1000), int(x2 * W / 1000), int(y2 * H / 1000)))


class RegionReader:
    """reader.read(image) by regions: the same fields as engine.Reader.read, so that glmocr/pdf.py and the CLI take either.
    The reader must have been made with max_tokens=MAX_TOKENS."""
    def __init__(self, reader, layout):
        self.reader, self.layout = reader, layout
        self._tmp = tempfile.TemporaryDirectory(prefix="glmocr_regions_")

    def read(self, image, scale=1.0, penalty=None, task="text"):
        from PIL import Image
        assert scale == 1.0 and penalty is None and task == "text", "a page read by regions takes the SDK's settings"
        Image.MAX_IMAGE_PIXELS = None
        img = (Image.open(image) if isinstance(image, (str, pathlib.Path)) else image).convert("RGB")
        path = pathlib.Path(self._tmp.name) / "page.png"; img.save(path)
        t = time.time(); found = self.layout.regions(path); read = []
        for r in found["regions"]:
            r = dict(r)
            if r["task_type"] in ("text", "table", "formula"):
                c = crop(img, r["bbox_2d"])
                if c.width > 1 and c.height > 1:
                    o = self.reader.read(c, penalty=PENALTY, task=r["task_type"])
                    r.update(content=o["text"], gen_tokens=o["gen_tokens"], prompt_tokens=o["prompt_tokens"], looped=o["looped"],
                             s=o["s"], peak_gb=o["peak_gb"])
            read.append(r)
        done = [r for r in read if "content" in r]
        gen = sum(r["gen_tokens"] for r in done)
        return {"text": self.layout.format(read), "s": time.time() - t, "layout_s": found["s"], "regions": len(done),
                "gen_tokens": max((r["gen_tokens"] for r in done), default=0), "gen_tokens_total": gen,
                "prompt_tokens": sum(r["prompt_tokens"] for r in done), "looped": any(r["looped"] for r in done), "loop_cut": None,
                "gen_tps": gen / max(sum(r["s"] for r in done), 1e-9), "peak_gb": max((r["peak_gb"] for r in done), default=0.0),
                "layout": {"sdk": self.layout.sdk, "keep": self.layout.keep, "regions": read}}
