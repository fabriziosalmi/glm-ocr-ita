"""work/gen_venv/bin/python generators/augment.py SRC OUT [--variants K] [--seed S] : K degraded copies of every page of the page set SRC
-> OUT, with the same text and entities (meta.json gets "aug" with what was applied). For training only: a degraded copy is not a
new page, and a set built from it is never used to measure.
Each copy draws one of three strengths and, at that strength, a random subset of photo and copy effects from Augraphy (folds,
shadow, uneven light, ink bleed, faded lines, paper noise, JPEG), a grey copy, focus blur, lost resolution, and a small perspective and rotation. The strongest level is
kept readable on purpose: a page degraded past reading would teach the model to invent text."""
import argparse, json, pathlib, random, shutil
import numpy as np
from PIL import Image
import augraphy as A

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--variants", type=int, default=3)
ap.add_argument("--seed", type=int, default=20261014); ap.add_argument("--max-side", type=int, default=1754)
a = ap.parse_args()
src, out = pathlib.Path(a.src), pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
meta = json.load(open(src / "meta.json", encoding="utf-8"))


def effects(level, rng):
    """Augraphy's photocopier and dirty-drum effects are left out: at any setting tried they erased whole lines of a form"""
    p = {"light": 0.25, "medium": 0.4, "heavy": 0.55}[level]
    pool = [("folding", lambda: A.Folding(fold_count=rng.randint(1, 2), fold_noise=0.0, gradient_width=(0.1, 0.2), gradient_height=(0.005, 0.01))),
            ("shadow", lambda: A.ShadowCast(shadow_side=rng.choice(["top", "bottom", "left", "right"]), shadow_opacity_range=(0.15, 0.4))),
            ("light", lambda: A.LightingGradient(max_brightness=255, min_brightness=rng.randint(150, 210))),
            ("inkbleed", lambda: A.InkBleed(intensity_range=(0.1, 0.2))),
            ("lowink", lambda: A.LowInkRandomLines(count_range=(2, 6))),
            ("noise", lambda: A.NoiseTexturize(sigma_range=(1, 3), turbulence_range=(2, 4))),
            ("jpeg", lambda: A.Jpeg(quality_range=(45, 70) if level == "heavy" else (60, 90)))]
    chosen = [(n, f) for n, f in pool if rng.random() < p] or [rng.choice(pool)]
    return [n for n, _ in chosen], A.AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=[f() for _, f in chosen])


def camera(img, level, rng):
    """what a phone or a copier adds and Augraphy does not: grey copy, focus blur, lost resolution"""
    from PIL import ImageFilter, ImageOps
    done = []
    if rng.random() < 0.3:
        img = ImageOps.autocontrast(img.convert("L"), cutoff=1).convert("RGB"); done.append("grey")
    if rng.random() < {"light": 0.3, "medium": 0.5, "heavy": 0.7}[level]:
        r = {"light": (0.4, 0.8), "medium": (0.6, 1.2), "heavy": (0.9, 1.6)}[level]
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*r))); done.append("blur")
    if rng.random() < 0.4:
        f = {"light": 0.85, "medium": 0.7, "heavy": 0.6}[level]
        small = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR); img = small.resize(img.size, Image.BILINEAR); done.append("lowres")
    return img, done


def geometry(img, level, rng):
    """a small rotation and a perspective, on a margin of paper-coloured background so that no text leaves the frame"""
    deg = {"light": 0.6, "medium": 1.5, "heavy": 3.0}[level] * rng.uniform(-1, 1)
    pad = int(0.04 * max(img.size))
    bg = tuple(int(c) for c in np.median(np.asarray(img).reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), bg); canvas.paste(img, (pad, pad))
    canvas = canvas.rotate(deg, resample=Image.BICUBIC, fillcolor=bg)
    w, h = canvas.size; s = {"light": 0.005, "medium": 0.015, "heavy": 0.03}[level]
    d = lambda: rng.uniform(0, s) * w
    quad = (d(), d(), d(), h - d(), w - d(), h - d(), w - d(), d())   # source corners: tl, bl, br, tr
    return canvas.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=bg), round(deg, 2)


rng = random.Random(a.seed); np.random.seed(a.seed)
new = []
for m in meta:
    img = Image.open(src / f"{m['id']}.png").convert("RGB")
    k = a.max_side / max(img.size)
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    for v in range(a.variants):
        level = rng.choice(["light", "medium", "medium", "heavy"])
        names, pipe = effects(level, rng)
        arr = pipe(np.asarray(img)[:, :, ::-1].copy())   # augraphy works in BGR
        im = Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]) if arr.ndim == 3 else arr).convert("RGB")
        im, cam = camera(im, level, rng); names += cam
        im, deg = geometry(im, level, rng)
        vid = f"{m['id']}_a{v}"
        im.save(out / f"{vid}.png"); shutil.copy(src / f"{m['id']}.txt", out / f"{vid}.txt")
        new.append({**m, "id": vid, "source": m["id"], "size": list(im.size), "aug": {"level": level, "effects": names, "rotate": deg}})
json.dump(new, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(new), "degraded pages from", len(meta), "->", out)
