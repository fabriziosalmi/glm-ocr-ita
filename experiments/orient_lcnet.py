"""work/layout_venv/bin/python experiments/orient_lcnet.py SET [SET ...]
PP-LCNet_x1_0_doc_ori (PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors, through transformers' pp_lcnet: no PaddleX) on page images.
SET is either work/orient_<set>/labels.json (experiments/orient_make.py: every page at 0/90/180/270, label counter-clockwise) or a
corpus name: then every page of the set as it is, labelled by its meta.json (rot90 -> 90, else 0).
Prints accuracy and the wrong answers; writes work/orient_lcnet_<name>.json (id -> predicted clockwise degrees, score).
The classifier answers the clockwise turn that straightens the page, so it is right when (pred + label) % 360 == 0."""
import json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from PIL import Image
from transformers import AutoImageProcessor, PPLCNetForImageClassification

ROOT = pathlib.Path(__file__).resolve().parent.parent
HUB = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
torch.set_num_threads(4)
PADDLE = "--hf-preprocessing" not in sys.argv; INK = "--ink" in sys.argv; sys.argv = [a for a in sys.argv if a != "--ink"]; sys.argv = [a for a in sys.argv if a != "--hf-preprocessing"]
proc = AutoImageProcessor.from_pretrained(HUB); model = PPLCNetForImageClassification.from_pretrained(HUB).eval()
labels = model.config.id2label


MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pixels(im):
    """PaddleX's own preprocessing: short side to 256 by bilinear interpolation without antialiasing (cv2.INTER_LINEAR), centre
    crop 224, BGR normalised with ImageNet's means in BGR order. transformers' processor resizes with antialiasing, which on a
    1754 px page brought down to 256 is another picture of the text: 150 of 160 on hard against PaddleX's 159 (2026-11-12)."""
    import numpy as np
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[[2, 1, 0]]   # BGR, as cv2 reads it
    h, w = x.shape[1:]; k = 256 / min(h, w)
    x = torch.nn.functional.interpolate(x[None], size=(round(h * k), round(w * k)), mode="bilinear", align_corners=False, antialias=False)[0]
    h, w = x.shape[1:]; t, l = (h - 224) // 2, (w - 224) // 2
    return ((x[:, t:t + 224, l:l + 224] - MEAN) / STD)[None]


def to_ink(im, margin=0.02):
    """The page cut to its ink (deskew._ink, the local-contrast mask), plus a margin: the centre crop then holds text, not the blank
    lower half of a sparse page (ocrbench_train: 15 of 50 upright pages taken for turned on the whole page, 2026-11-12)."""
    import numpy as np
    from glmocr import deskew
    g = im.convert("L"); k = min(1.0, 800 / max(g.size)); small = g.resize((round(g.width * k), round(g.height * k)))
    ys, xs = np.nonzero(deskew._ink(np.asarray(small)))
    if len(ys) < 50:
        return im
    m = margin * max(small.size)
    box = [max(0, xs.min() - m), max(0, ys.min() - m), min(small.width, xs.max() + m), min(small.height, ys.max() + m)]
    return im.crop(tuple(round(v / k) for v in box))


def predict(path):
    im = Image.open(path).convert("RGB")
    if INK:
        im = to_ink(im)
    with torch.no_grad():
        p = model(pixel_values=pixels(im) if PADDLE else proc(images=im, return_tensors="pt")["pixel_values"]).last_hidden_state.softmax(-1)[0]
    k = int(p.argmax())
    return int(labels[k]), float(p[k])


for arg in sys.argv[1:]:
    if arg.endswith(".json"):
        rows = [{"id": pathlib.Path(r["file"]).stem, "file": ROOT / r["file"], "label": r["label"]} for r in json.load(open(ROOT / arg))]
        name = pathlib.Path(arg).parent.name
    else:
        sys.path.insert(0, str(ROOT)); from glmocr import corpus
        d = corpus.corpus_dir(arg)
        rows = [{"id": m["id"], "file": d / f"{m['id']}.png", "label": 90 if m.get("rot90") else 0} for m in corpus.load_meta(arg)]
        name = arg.strip("/").replace("/", "_")
    t = time.time(); out = {}
    for r in rows:
        out[r["id"]] = dict(zip(("pred", "score"), predict(r["file"])))
    ms = 1000 * (time.time() - t) / len(rows)
    ok = sum((out[r["id"]]["pred"] + r["label"]) % 360 == 0 for r in rows)
    print(f"{name}: {ok}/{len(rows)} right, {ms:.0f} ms/page, labels {labels}")
    for r in rows:
        o = out[r["id"]]
        if (o["pred"] + r["label"]) % 360:
            print(f"   wrong: {r['id']} label {r['label']} pred {o['pred']} score {o['score']:.2f}")
    json.dump(out, open(ROOT / f"work/orient_lcnet_{name}{'_ink' if INK else ''}.json", "w"), indent=0)
