"""work/gen_venv/bin/python generators/augment.py SRC OUT [--variants K] [--seed S] : K degraded copies of every page of the page set SRC
-> OUT, with the same text and entities (meta.json gets "aug": the severity, every effect and its strength). For training only: a
degraded copy is not a new page, and a set built from it is never used to measure.
SEVERITY IS CONTINUOUS. Each copy draws a severity s in [0, 1] (Beta(2, 2.2): centred a little below the middle, with long tails
towards clean and towards hard), and every effect follows s smoothly: its probability, and its strength through a value drawn
around s for that effect alone. No two copies share a strength, and since training makes one pass, every degraded page is seen
once. Discrete levels were tried first and dropped: three families of damage are something a model can learn by heart.
Effects: from Augraphy, folds, shadow, uneven light, ink bleed, faded lines, paper noise, JPEG; added here, a grey copy, focus
blur, lost resolution, rotation and perspective. Augraphy's photocopier and dirty-drum effects are left out: at every setting
tried they erased whole lines. The caps (blur radius 1.0 px, resolution down to 72%, JPEG quality 55 at 1754 px) were set by eye so that the smallest print of
the F24, 6-point labels, stays readable at s = 1: a page degraded past reading would teach the model to invent text."""
import argparse, json, pathlib, random, shutil
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import augraphy as A

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--variants", type=int, default=3)
ap.add_argument("--seed", type=int, default=20261014); ap.add_argument("--max-side", type=int, default=1754)
a = ap.parse_args()
src, out = pathlib.Path(a.src), pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
meta = json.load(open(src / "meta.json", encoding="utf-8"))
lerp = lambda lo, hi, t: lo + (hi - lo) * min(1.0, max(0.0, t))


def strength(s, rng):
    """this effect's own strength: around the page's severity, never the same twice"""
    return min(1.0, max(0.0, rng.gauss(s, 0.15)))


def degrade(img, s, rng):
    done = {}
    use = lambda base: rng.random() < lerp(base, base + 0.45, s)   # every effect grows more likely as s grows
    post = []
    if use(0.10):
        t = strength(s, rng); done["folding"] = round(t, 2)
        post.append(A.Folding(fold_count=1 + int(t * 2.5), fold_noise=0.0, gradient_width=(0.1, 0.2), gradient_height=(0.003, lerp(0.005, 0.015, t))))
    if use(0.10):
        t = strength(s, rng); done["shadow"] = round(t, 2)
        o = lerp(0.1, 0.45, t); post.append(A.ShadowCast(shadow_side=rng.choice(["top", "bottom", "left", "right"]), shadow_opacity_range=(o, o + 0.05)))
    if use(0.15):
        t = strength(s, rng); done["light"] = round(t, 2)
        post.append(A.LightingGradient(max_brightness=255, min_brightness=int(lerp(220, 140, t))))
    if use(0.10):
        t = strength(s, rng); done["inkbleed"] = round(t, 2)
        i = lerp(0.05, 0.18, t); post.append(A.InkBleed(intensity_range=(i, i + 0.03)))
    if use(0.10):
        t = strength(s, rng); done["lowink"] = round(t, 2)
        c = int(lerp(1, 8, t)); post.append(A.LowInkRandomLines(count_range=(c, c + 1)))
    if use(0.15):
        t = strength(s, rng); done["noise"] = round(t, 2)
        g = int(round(lerp(1, 4, t))); post.append(A.NoiseTexturize(sigma_range=(g, g + 1), turbulence_range=(2, 4)))   # Augraphy takes whole numbers here
    if use(0.25):
        t = strength(s, rng); done["jpeg"] = round(t, 2)
        q = int(lerp(92, 55, t)); post.append(A.Jpeg(quality_range=(q, q + 2)))
    if post:
        arr = A.AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=post)(np.asarray(img)[:, :, ::-1].copy())   # BGR
        img = Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]) if arr.ndim == 3 else arr).convert("RGB")
    if use(0.10):
        img = ImageOps.autocontrast(img.convert("L"), cutoff=1).convert("RGB"); done["grey"] = 1
    if use(0.20):
        t = strength(s, rng); r = lerp(0.3, 1.0, t); done["blur"] = round(r, 2)
        img = img.filter(ImageFilter.GaussianBlur(r))
    if use(0.15):
        t = strength(s, rng); f = lerp(0.92, 0.72, t); done["lowres"] = round(f, 2)
        img = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR).resize(img.size, Image.BILINEAR)
    return geometry(img, s, rng, done)


def geometry(img, s, rng, done):
    """rotation and perspective growing with s, on a margin of paper-coloured background so that no text leaves the frame"""
    deg = lerp(0.2, 3.2, strength(s, rng)) * rng.choice([-1, 1])
    pad = int(0.04 * max(img.size))
    bg = tuple(int(c) for c in np.median(np.asarray(img).reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), bg); canvas.paste(img, (pad, pad))
    canvas = canvas.rotate(deg, resample=Image.BICUBIC, fillcolor=bg)
    w, h = canvas.size; p = lerp(0.002, 0.03, strength(s, rng))
    d = lambda: rng.uniform(0, p) * w
    quad = (d(), d(), d(), h - d(), w - d(), h - d(), w - d(), d())   # source corners: tl, bl, br, tr
    done["rotate"] = round(deg, 2); done["perspective"] = round(p, 4)
    return canvas.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=bg), done


rng = random.Random(a.seed); np.random.seed(a.seed)
new = []
for m in meta:
    img = Image.open(src / f"{m['id']}.png").convert("RGB")
    k = a.max_side / max(img.size)
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    for v in range(a.variants):
        s = rng.betavariate(2, 2.2)
        im, done = degrade(img, s, rng)
        vid = f"{m['id']}_a{v}"
        im.save(out / f"{vid}.png"); shutil.copy(src / f"{m['id']}.txt", out / f"{vid}.txt")
        new.append({**m, "id": vid, "source": m["id"], "size": list(im.size), "aug": {"severity": round(s, 3), **done}})
json.dump(new, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(new), "degraded pages from", len(meta), "->", out)
