"""Straighten a skewed page before it is read: the angle by projection profile, as stage 02 of the owner's ocr-preprocessing
harness does it. The rotation that makes the rows of ink sharpest is the one that undoes the skew. numpy and PIL only.

Only rotation: the preprocessing probe of 2026-10-17 (docs/ROADMAP.md) found every stage that rewrites grey levels (binarization,
division) worse for this model, and rotation better where a page is really skewed."""
import numpy as np
from PIL import Image, ImageFilter

WIDTH = 800          # the angle is searched on a copy this wide: enough rows of text, a fraction of a second
SPAN = 20.0          # degrees searched either way; a page turned by 90 is another problem (rot90 pages in the corpora)


def _ink(gray, size=15, dark=0.8):
    """True where a pixel is darker than `dark` times the page around it. The page is the local maximum, smoothed: a global
    threshold takes a photo's shadow for ink (ocrbench_train doc_082, half the page), and the shadow's edge for a line."""
    g = Image.fromarray(gray)
    bg = np.asarray(g.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.BoxBlur(size)), dtype=np.float32)
    return gray < dark * bg


def _sharpness(mask, deg):
    rows = np.asarray(mask.rotate(deg, resample=Image.NEAREST, fillcolor=0), dtype=np.float32).sum(axis=1)
    return float(np.sum(np.diff(rows) ** 2))


def angle(image):
    """Degrees the page is turned counterclockwise (PIL's sense): rotate by minus this to straighten it. 0.05 degree steps."""
    g = image.convert("L")
    if g.width > WIDTH:
        g = g.resize((WIDTH, round(g.height * WIDTH / g.width)), Image.BILINEAR)
    mask = Image.fromarray((_ink(np.asarray(g)) * 255).astype(np.uint8))
    best = max(np.arange(-SPAN, SPAN + 1e-9, 0.5), key=lambda d: _sharpness(mask, d))
    best = max(np.arange(best - 0.5, best + 0.5 + 1e-9, 0.05), key=lambda d: _sharpness(mask, d))
    return -round(float(best), 2)


def straighten(image, min_deg=1.0, trim=False):
    """(image, detected angle). Rotated only when the angle is at least min_deg, otherwise the same object comes back.
    The canvas grows to keep every corner, filled with the page's border colour. trim: crop to the ink plus a 2% margin
    afterwards, so that the larger canvas does not cost image tokens."""
    a = angle(image)
    if abs(a) < min_deg:
        return image, a
    rgb = image.convert("RGB"); arr = np.asarray(rgb)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    out = rgb.rotate(-a, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    if trim:
        ink = _ink(np.asarray(out.convert("L")))
        ys, xs = np.nonzero(ink)
        if len(ys):
            m = round(0.02 * max(out.size))
            out = out.crop((max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m + 1, out.width), min(ys.max() + m + 1, out.height)))
    return out, a
