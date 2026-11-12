"""Bands: the top 16% and the bottom 14% of a page read again on their own, and their lines that the whole-page reading lacks put
into it. Where the whole page loses its letterhead or its closing lines (a closing under blank space, a footer), the band holds
them. Measured in docs/LEDGER.md ("Bands, round 2" and "The guard for turned pages", 2026-11-12): on 200 letters no model had
read, +3.69 [+2.67, +4.83] entity recall, no page lost, CER 3.0% to 3.3%, +20% time; on the dev sets never worse.

No band is read on a page the orientation classifier is sure is turned (glmocr/orient.py): a band across a page on its side holds
pieces of every line. The merge was written in experiments/pieces.py, where the other ways of cutting a page are still measured,
and moved here unchanged."""
import re
import sys
import time
import unicodedata

from rapidfuzz import fuzz

from . import guard, metrics

BANDS = {"top": (0.0, 0.16), "bottom": (0.86, 1.0)}   # share of the page's height
SAME, CLOSE, MIN_LINE = 90, 75, 6   # line similarity (rapidfuzz ratio on squashed text): present, close enough to replace; shortest line
VARIANT, COVER = 60, 0.8   # a piece line this close to a line of the page, or with this share of its words on the page, is there
OVERLAP = {"top": "half1", "half1": "half2", "half2": "half1", "bottom": "half2"}   # the piece that holds a cut edge line whole


def lines(t):
    return [l for l in t.split("\n") if l.strip()]


def inside(s, longer):
    """s (squashed) appears, nearly, inside the squashed text `longer`: only when s is the shorter one."""
    return len(s) <= len(longer) and fuzz.partial_ratio(s, longer) >= SAME


def words(t):
    return set(re.findall(r"\w{4,}", unicodedata.normalize("NFKC", t).lower()))


# What a band returns that is not text of the page (round 2, 2026-11-12, written on the dev sets' band readings): a band that frames
# no text is described, not read ("This image is too blurry to recognize any text."), a band is sometimes wrapped in a markdown
# fence, and once a blank band came back as a count ("1. 2. 3. ... 30.").
FENCE = re.compile(r"^\s*```")
DESCRIBES = re.compile(r"(?i)^\s*(?:(?:this|the)\s+(?:image|picture|photo(?:graph)?)\b|this is an? (?:sample|image|picture|photo)\b"
                       r"|i'?m sorry|i can(?:not|'t)\b|unable to\b|there is no (?:visible |readable )?text"
                       r"|l'immagine\b|questa immagine\b|non è possibile\b|non c'è (?:alcun )?testo)")


def counting(l):
    """a line that is nothing but 1. 2. 3. ... : five numbers or more, each one more than the last"""
    n = [int(x) for x in re.findall(r"\d+", l)]
    return bool(re.fullmatch(r"[\d.)\s]+", l)) and len(n) >= 5 and all(b == a + 1 for a, b in zip(n, n[1:]))


def not_text(l):
    """a line of a piece's reading that is not text of the page: a fence, a description of the image, a count"""
    return bool(FENCE.match(l) or DESCRIBES.match(l) or counting(l))


def merge(whole, pieces, replace=False, use=("top", "half1", "half2", "bottom"), clean=False):
    """pieces: name -> {"text": ...}. Lines are compared on letters and digits only (a markdown table and its plain reading match),
    and a piece line is not added when it is a variant of a line of the page (ratio >= VARIANT: a degraded band reads the same line
    worse), when COVER of its words of 4 letters or more are on the page (a band across two columns joins two lines; a line the page
    wraps), or when it is nearly inside the page's text. A line at a piece's cut edge is not added when the overlapping piece holds
    it whole. clean: the piece's lines that are not text of the page (not_text) are dropped before anything else.
    The reader uses use=("top", "bottom"), clean=True (merge_bands)."""
    out = lines(whole); keys = [metrics.alnum(l) for l in out]
    for name in use:
        pl = [l for l in lines(pieces[name]["text"]) if not (clean and not_text(l))]
        edge = {"top": {len(pl) - 1}, "half1": {len(pl) - 1}, "half2": {0}, "bottom": {0}}[name]
        other = [metrics.alnum(l) for l in lines(pieces[OVERLAP[name]]["text"])] if OVERLAP[name] in use else []
        prev = -1 if name in ("top", "half1") else None
        for n, l in enumerate(pl):
            s = metrics.alnum(l)
            if len(s) < MIN_LINE:
                continue
            best_i, best_r = max(((i, fuzz.ratio(s, k)) for i, k in enumerate(keys)), key=lambda x: x[1], default=(None, 0))
            if replace and name.startswith("half") and CLOSE <= best_r < 100 and n not in edge:
                out[best_i], keys[best_i] = l, s; prev = best_i; continue
            w = words(l); page_words = words("\n".join(out))
            if best_r >= VARIANT or (len(w) >= 2 and len(w & page_words) >= COVER * len(w)) or inside(s, "".join(keys)):
                prev = best_i; continue
            if n in edge and any(len(o) > len(s) and inside(s, o) for o in other):
                continue
            pos = len(out) if prev is None else prev + 1
            out.insert(pos, l); keys.insert(pos, s); prev = pos
    return "\n".join(out)


def merge_bands(whole, bands):
    """What the reader returns: the whole page with the top and bottom bands merged in, lines that are not text dropped."""
    return merge(whole, bands, use=("top", "bottom"), clean=True)


def crop(page, name):
    """The band `name` of a PIL page, full width."""
    y0, y1 = BANDS[name]
    return page.crop((0, round(y0 * page.height), page.width, round(y1 * page.height)))


class BandReader:
    """A reader (engine.Reader or llama.LlamaReader) whose read() also reads the two bands and merges them in. The page the bands are
    cut from is the one the model reads: upright as its EXIF says. With --deskew the bands are not read (not measured together)."""
    FIELDS = ("text", "s", "prompt_tokens", "gen_tokens", "loop_cut")

    def __init__(self, reader):
        self.reader = reader

    def __getattr__(self, name):   # load_s, build_info, server_args, ...
        return getattr(self.reader, name)

    def read(self, image, scale=1.0, deskew=None, trim=False, **kw):
        from . import orient
        from .engine import open_page
        if deskew is not None:
            print("bands: not read with --deskew (not measured together)", file=sys.stderr)
            out = self.reader.read(image, scale, deskew=deskew, trim=trim, **kw); self.last = out
            return out
        whole = self.reader.read(image, scale, **kw)
        out = dict(whole); self.last = out
        t = time.time()
        page = open_page(image).convert("RGB")
        deg, score = orient.classify(page)
        t = time.time() - t   # the classifier's own time; each band adds the time its reader reports, as experiments/pieces.py counts
        rec = {"orientation": deg, "score": round(score, 4)}
        out["bands"] = rec
        if deg != 0 and score >= orient.SURE:
            rec["skipped"] = "turned"; out["s"] = whole["s"] + t
            return out
        read, peak = {}, whole.get("peak_gb", 0)
        try:
            for name in BANDS:
                r = self.reader.read(crop(page, name), scale, **kw)
                read[name] = rec[name] = {k: r.get(k) for k in self.FIELDS} | {"retry": bool(r.get("retry"))}
                peak = max(peak, r.get("peak_gb", 0))
        except guard.MemoryStop:   # the page itself was read: keep it as the whole page alone, then stop
            rec["stopped"] = True
            raise
        out["whole_text"] = whole["text"]
        out["text"] = merge_bands(whole["text"], read)
        out["s"] = whole["s"] + t + sum(r["s"] for r in read.values())
        out["peak_gb"] = peak
        return out
