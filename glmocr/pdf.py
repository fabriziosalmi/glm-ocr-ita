"""PDF pages: the text layer where it can be trusted, the model elsewhere.

A digital PDF already holds its text. Where that text layer is sound, reading it is exact and takes milliseconds; the model is
needed only for scans and for pages whose layer fails a check. Modes, per page:
  auto  the text layer if it passes every check below, else OCR of the page rendered at 150 dpi (default)
  ocr   always the model
  text  always the text layer, even when a check fails (and nothing for a page that has none)

The checks and the run extraction are adapted from Indra OCR, datalab-to/indra, indra_ocr/textlayer.py (Apache-2.0,
Datalab; NOTICE and licenses/), 2026-10-15. Changes: the order check is part of the verdict (Indra also asks an order model, which
we do not have, so two columns written line by line across the gutter are not caught here); a run's character boxes are not kept;
the text of a page is its runs in content order, one per line. A layer is not trusted when:
- more than 5% of its text objects are drawn invisibly: an OCR layer laid over a scan, as good as the OCR that made it;
- more than 0.5% of its characters are replacement, private-use or control characters: a broken font map;
- on a page whose letters are 90% Latin or more, three letters or more of another script than Latin, Greek or Cyrillic: a font
  map that is broken without leaving a bad character (added here; the Gazzetta Ufficiale sets its running head in small caps
  whose codes come out as Malayalam and Sinhala letters, "Gൺඓඓൾඍඍൺ Uൿൿංർංൺඅൾ", on 5 of 313 pages sampled on 2026-10-15);
- more than 3% of its runs have no ink under them in the render: hidden or misplaced text;
- its content order moves upward, other than to a column on the right, more than max(1, 2% of the runs): not a reading order.
Text drawn as paths or inside images is not in the layer at all and is not seen by the checks: a scanned stamp on a digital page
is lost in auto mode, as it is in the ground truth of most page sets built from text layers.
"""
import math
import time
import unicodedata

import numpy as np

DPI_LONG_SIDE = 1754              # the render the model reads: A4 at 150 dpi, as every page set of this repository
MAX_INVISIBLE_FRAC = 0.05
MAX_BAD_CHAR_FRAC = 0.005
MAX_NO_INK_RUN_FRAC = 0.03
MAX_ALIEN_LETTERS = 3
DARK = 160                        # grey level below which a pixel counts as ink
MODES = ("auto", "ocr", "text")


def bad_char_count(text):
    return sum(c == "�" or unicodedata.category(c) in ("Co", "Cs", "Cn") or (unicodedata.category(c) == "Cc" and c != "\t")
               for c in text)


def _script(c):
    try:
        return unicodedata.name(c).split()[0]
    except ValueError:
        return "?"


def alien_letters(text):
    """Letters of another script than Latin, Greek or Cyrillic, counted only when the text's letters are 90% Latin or more."""
    letters = [_script(c) for c in text if c.isalpha()]
    if not letters or letters.count("LATIN") < 0.9 * len(letters):
        return 0
    return sum(s not in ("LATIN", "GREEK", "CYRILLIC") for s in letters)


def _device_affine(page):
    """[2, 3] map from PDF page space to the rendered frame (points, top-left origin), from pdfium's FPDF_PageToDevice."""
    import ctypes
    import pypdfium2.raw as R
    W, H = page.get_size(); S = 1000

    def dev(x, y):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        R.FPDF_PageToDevice(page.raw, 0, 0, round(W * S), round(H * S), 0, x, y, dx, dy)
        return np.array([dx.value, dy.value], float) / S
    o, ex, ey = dev(0.0, 0.0), dev(1.0, 0.0), dev(0.0, 1.0)
    return np.stack([ex - o, ey - o, o], 1)


def extract_runs(page):
    """([(box, text)], stats): the upright text runs of a pypdfium2 page in content order, boxes in points of the rendered frame.
    A run is pdfium's text between the line breaks it generates; characters not upright on the rendered page break a run and are
    dropped, as are characters outside the rendered frame."""
    import ctypes
    import pypdfium2.raw as R
    tp = page.get_textpage()
    try:
        L = _device_affine(page)[:, :2]
        raw_tp = tp.raw; bl, br, bb, bt = (ctypes.c_double() for _ in range(4))
        upright, chars, boxes, breaks = {}, [], [], []
        n_chars = n_rot = n_bad = 0
        for i in range(tp.count_chars()):
            c = chr(R.FPDFText_GetUnicode(raw_tp, i))
            if c in "\r\n":
                breaks.append(len(chars)); continue
            if c == "\x02":   # pdfium's mark for a hyphen at a line break
                c = "-"
            if not c.isspace():
                n_chars += 1; n_bad += bad_char_count(c)
                a = R.FPDFText_GetCharAngle(raw_tp, i)
                if a not in upright:
                    d = L @ np.array([math.cos(a), math.sin(a)])
                    upright[a] = not (d[0] <= 0 or abs(d[1]) > 1e-3 * max(abs(d[0]), 1e-9))
                if not upright[a]:
                    n_rot += 1; breaks.append(len(chars)); continue
            chars.append(c)
            R.FPDFText_GetCharBox(raw_tp, i, ctypes.byref(bl), ctypes.byref(br), ctypes.byref(bb), ctypes.byref(bt))
            boxes.append((bl.value, bb.value, br.value, bt.value))
    finally:
        tp.close()
    stats = {"chars": n_chars, "rotated": n_rot, "bad": n_bad}
    if not chars:
        return [], stats
    A = _device_affine(page); b = np.asarray(boxes, float)
    xs, ys = b[:, [0, 2, 0, 2]], b[:, [1, 1, 3, 3]]
    px, py = A[0, 0] * xs + A[0, 1] * ys + A[0, 2], A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    box = np.stack([px.min(1), py.min(1), px.max(1), py.max(1)], 1)
    W, H = page.get_size(); cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    visible = (cx >= 0) & (cx <= W) & (cy >= 0) & (cy <= H); stats["off_page"] = int((~visible).sum())
    runs, start = [], 0
    for end in sorted(set(breaks + [len(chars)])):
        seg = [i for i in range(start, end) if visible[i]]; start = end
        ink = [i for i in seg if not chars[i].isspace()]
        if ink:
            runs.append((tuple(np.r_[box[ink, :2].min(0), box[ink, 2:].max(0)]), " ".join("".join(chars[i] for i in seg).split())))
    return runs, stats


def invisible_text_frac(page):
    import pypdfium2.raw as R
    n = inv = 0
    for obj in page.get_objects(max_depth=4):
        if obj.type == R.FPDF_PAGEOBJ_TEXT:
            n += 1; inv += R.FPDFTextObj_GetTextRenderMode(obj.raw) == R.FPDF_TEXTRENDERMODE_INVISIBLE
    return inv / n if n else 0.0


def no_ink_frac(gray, runs, scale):
    """Share of runs whose box holds no dark pixel in gray, rendered at scale pixels per point."""
    ink = gray < DARK; H, W = ink.shape; empty = 0
    for (x0, y0, x1, y1), _ in runs:
        empty += not ink[max(0, int(y0 * scale)):min(H, int(y1 * scale) + 1), max(0, int(x0 * scale)):min(W, int(x1 * scale) + 1)].any()
    return empty / max(len(runs), 1)


def order_backtracks(boxes):
    """Moves upward in content order that do not go to a column on the right: reading goes down a column, then up to the top of
    the next column on the right, and any other upward move means the content order is not a reading order."""
    boxes = np.asarray(boxes, float)
    if len(boxes) < 3:
        return 0
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    return sum(b[1] < a[1] - 0.5 * h and b[0] < a[2] - h for a, b in zip(boxes[:-1], boxes[1:]))


def check_layer(page, gray, scale):
    """(runs, reason, stats): reason is "" when the layer can be trusted."""
    inv = invisible_text_frac(page)
    if inv > MAX_INVISIBLE_FRAC:
        return [], f"invisible text {inv:.0%} (an OCR layer)", {}
    runs, st = extract_runs(page)
    if not runs:
        return [], "no text layer", st
    if st["bad"] > MAX_BAD_CHAR_FRAC * max(st["chars"], 1):
        return runs, "broken font map", st
    st["alien"] = alien_letters(" ".join(t for _, t in runs))
    if st["alien"] >= MAX_ALIEN_LETTERS:
        return runs, f"broken font map ({st['alien']} letters of another script)", st
    st["no_ink"] = round(no_ink_frac(gray, runs, scale), 4)
    if st["no_ink"] > MAX_NO_INK_RUN_FRAC:
        return runs, f"{st['no_ink']:.0%} of text runs have no ink", st
    st["backtracks"] = int(order_backtracks([b for b, _ in runs]))
    if st["backtracks"] > max(1, int(0.02 * len(runs))):
        return runs, f"content order is not a reading order ({st['backtracks']} moves back)", st
    return runs, "", st


def page_ranges(spec, n):
    """'1-3,7' -> [0, 1, 2, 6] (pages counted from 1, as printed); None -> every page."""
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a) - 1, int(b or a))
    return [i for i in out if 0 <= i < n]


def read_pdf(path, reader=None, mode="auto", pages=None, make_reader=None):
    """Yield one dict per page: page (from 1), source ("text" or "ocr"), reason (why the layer was not used, if it was not),
    text, s, and the model's own fields when it read the page. The model is loaded only when a page needs it: make_reader()
    is called then, once, unless a reader is given."""
    import pypdfium2 as pdfium
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    pdf = pdfium.PdfDocument(str(path))
    for i in page_ranges(pages, len(pdf)):
        t = time.time(); page = pdf[i]
        k = DPI_LONG_SIDE / max(page.get_size())
        img = page.render(scale=k).to_pil().convert("RGB")
        runs, reason, st = check_layer(page, np.asarray(img.convert("L")), k) if mode != "ocr" else ([], "mode ocr", {})
        if mode == "text" or (mode == "auto" and not reason):
            yield {"page": i + 1, "source": "text", "reason": reason, "text": "\n".join(s for _, s in runs), "s": time.time() - t, "layer": st}
            continue
        if reader is None:
            reader = make_reader()
        r = reader.read(img)
        yield {**r, "page": i + 1, "source": "ocr", "reason": reason, "s": time.time() - t, "layer": st}
