"""python experiments/pieces.py read BASE_TAG --corpus NAME [--wait] [--limit N]
   python experiments/pieces.py score BASE_TAG --corpus NAME
Does reading a page again in pieces, and merging the pieces into the whole-page reading, recover what the whole page loses,
and at what cost in time and in text that is not on the page? No training, no ground truth in the reading or the merge.

read   for every page of BASE_TAG (a whole-page run of the reader, with its retry), read four pieces with the model of BASE_TAG
       and the same retry: the top 16% and the bottom 14% of the page as they are (letterhead and footer), and two overlapping
       halves (0-56%, 44-100%) enlarged 1.4x (small print). -> results/BASE_TAG_pieces.json, resumes if interrupted.
score  no model. For each policy: entity recall with a paired bootstrap against the whole page, CER, how much text the merged
       reading holds against the reference (excess), pages gained and lost, seconds per page. Policies:
         whole       BASE_TAG as it is
         union       whole + every piece joined: an upper bound on recall only, never a reading (it repeats the page)
         insert      the whole page, plus the lines of the pieces that are not in it, put after the line they follow in the piece
         replace     insert, and a line of an enlarged half that is close to a line of the whole page (ratio 75 to 99) takes its place:
                     the bet that enlarged small print is read better, measured here, not assumed
       and the same merges restricted to pieces that pass a trigger (gated), to see what each trigger costs and saves.
Lines at the cut edge of a piece (the last of the top band and of the first half, the first of the second half and of the bottom
band) are never inserted: they are cut through, and the overlapping piece holds them whole."""
import argparse, json, re, statistics, sys, unicodedata

from rapidfuzz import fuzz

from PIL import Image

from glmocr import corpus, deskew, metrics

PIECES = {"top": (0.0, 0.16, 1.0), "half1": (0.0, 0.56, 1.4), "half2": (0.44, 1.0, 1.4), "bottom": (0.86, 1.0, 1.0)}
SAME, CLOSE, MIN_LINE = 90, 75, 6   # line similarity (rapidfuzz ratio on squashed text): present, close enough to replace; shortest line


def lines(t):
    return [l for l in t.split("\n") if l.strip()]


OVERLAP = {"top": "half1", "half1": "half2", "half2": "half1", "bottom": "half2"}   # the piece that holds a cut edge line whole


def inside(s, longer):
    """s (squashed) appears, nearly, inside the squashed text `longer`: only when s is the shorter one."""
    return len(s) <= len(longer) and fuzz.partial_ratio(s, longer) >= SAME


TURNED = 1.0   # deskew.turn_ratio above this: no bands. Separates every page of hard/ and held/ (upright at most 0.97, turned
               # 1.07 or more); on real photos it also takes grid paper and charts for turned pages, which only costs their bands
VARIANT, COVER = 60, 0.8   # v2: a piece line this close to a line of the page, or with this share of its words on the page, is there


def words(t):
    return set(re.findall(r"\w{4,}", unicodedata.normalize("NFKC", t).lower()))


# Round 2 (2026-11-12, written on the dev sets' band readings; docs/ROADMAP.md, "After v1.3.0", item 8): what a band returns that is
# not text of the page. A band that frames no text is described, not read ("This image is too blurry to recognize any text."), a
# band is sometimes wrapped in a markdown fence, and once a blank band came back as a count ("1. 2. 3. ... 30.").
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


def merge(whole, pieces, replace=False, use=("top", "half1", "half2", "bottom"), v2=True, clean=False):
    """v2 (2026-11-11, chosen on the four dev sets after v1 duplicated lines there): lines are compared on letters and digits only
    (a markdown table and its plain reading match), and a piece line is not added when it is a variant of a line of the page
    (ratio >= VARIANT: a degraded band reads the same line worse), when COVER of its words of 4 letters or more are on the page
    (a band across two columns joins two lines; a line the page wraps), or when it is nearly inside the page's text.
    clean (round 2): the piece's lines that are not text of the page (not_text) are dropped before anything else."""
    if not v2:
        return merge_v1(whole, pieces, replace, use)
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


def merge_v1(whole, pieces, replace=False, use=("top", "half1", "half2", "bottom")):
    out = lines(whole); keys = [metrics.squash(l) for l in out]
    for name in use:
        pl = lines(pieces[name]["text"])
        edge = {"top": {len(pl) - 1}, "half1": {len(pl) - 1}, "half2": {0}, "bottom": {0}}[name]
        other = [metrics.squash(l) for l in lines(pieces[OVERLAP[name]]["text"])] if OVERLAP[name] in use else []
        prev = -1 if name in ("top", "half1") else None
        for n, l in enumerate(pl):
            s = metrics.squash(l)
            if len(s) < MIN_LINE:
                continue
            cut = n in edge and any(len(o) > len(s) and inside(s, o) for o in other)   # a fragment the overlapping piece holds whole
            best_i, best_r = max(((i, fuzz.ratio(s, k)) for i, k in enumerate(keys)), key=lambda x: x[1], default=(None, 0))
            if replace and name.startswith("half") and CLOSE <= best_r < 100 and n not in edge:
                out[best_i], keys[best_i] = l, s; prev = best_i; continue   # the enlarged half's reading of the same line wins
            near = "".join(keys[max(0, best_i - 3):best_i + 4]) if keys else ""
            if best_r >= SAME or inside(s, near):
                prev = best_i; continue
            if cut:
                continue
            pos = len(out) if prev is None else prev + 1
            out.insert(pos, l); keys.insert(pos, s); prev = pos
    return "\n".join(out)


def cmd_read(a):
    from PIL import Image
    from glmocr.engine import Reader
    base = corpus.load_run(a.base)
    d = corpus.corpus_dir(a.corpus)
    out = corpus.result_path(f"{a.base}_pieces")
    res = json.load(open(out)) if out.exists() else {"base": a.base, "corpus": a.corpus, "pieces": PIECES, "pages": {}}
    ids = [i for i in base["pages"] if not set(a.only) <= set(res["pages"].get(i, {}))][: a.limit]
    if not ids:
        print("nothing to read"); return
    reader = Reader(base["model"], max_tokens=base.get("max_tokens", 3000), wait=a.wait, prompt_style=base.get("prompt", "official"),
                    retry_penalty=base.get("retry_penalty"), mem_gb=a.mem_gb)
    for i in ids:
        im = Image.open(d / f"{i}.png").convert("RGB"); w, h = im.size
        page = dict(res["pages"].get(i, {}))
        for name, (y0, y1, z) in PIECES.items():
            if name not in a.only or name in page:
                continue
            c = im.crop((0, round(y0 * h), w, round(y1 * h)))
            if z != 1.0:
                c = c.resize((round(c.width * z), round(c.height * z)), Image.LANCZOS)
            r = reader.read(c)
            page[name] = {"text": r["text"], "s": r["s"], "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"],
                          "loop_cut": r.get("loop_cut"), "retry": bool(r.get("retry"))}
        res["pages"][i] = page
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        print(i, {k: round(v["s"], 1) for k, v in page.items()}, flush=True)
    print("->", out)


def cmd_score(a):
    base = corpus.load_run(a.base)["pages"]
    pcs = corpus.load_run(f"{a.base}_pieces")["pages"]
    have = set.intersection(*(set(v) for v in pcs.values())) if pcs else set()
    skip = set(open(a.exclude).read().split()) if a.exclude else set()
    meta = [m for m in corpus.load_meta(a.corpus) if m["id"] in pcs and m["id"] not in skip]
    ids = [m["id"] for m in meta]
    allp = ("top", "half1", "half2", "bottom")
    d = corpus.corpus_dir(a.corpus)
    turned = {i: deskew.turn_ratio(Image.open(d / f"{i}.png")) > TURNED for i in ids}   # no bands on a page that may be on its side
    print(f"pages taken as turned (turn ratio above {TURNED}): {sum(turned.values())} of {len(ids)}, of them labelled rot90: "
          f"{sum(turned[m['id']] and bool(m.get('rot90')) for m in meta)}; labelled rot90 in all: {sum(bool(m.get('rot90')) for m in meta)}")
    ori = json.load(open(a.orient)) if a.orient else {}   # experiments/orient_lcnet.py: id -> clockwise degrees and score
    if ori and set(ids) - set(ori):
        sys.exit(f"{a.orient} has no answer for {len(set(ids) - set(ori))} pages")
    sure = {i: ori[i]["pred"] != 0 and ori[i]["score"] >= a.orient_min for i in ori}   # turned, by the classifier, and sure of it
    if ori:
        print(f"pages the classifier is sure are turned (score {a.orient_min} or more): {sum(sure[i] for i in ids)} of {len(ids)}, "
              f"of them labelled rot90: {sum(sure[m['id']] and bool(m.get('rot90')) for m in meta)}")
    pol = {"whole": lambda i: base[i]["text"],
           "union": lambda i: "\n".join([base[i]["text"]] + [pcs[i][k]["text"] for k in allp if k in pcs[i]]),
           "insert": lambda i: merge(base[i]["text"], pcs[i]),
           "replace": lambda i: merge(base[i]["text"], pcs[i], replace=True),
           "insert bands": lambda i: merge(base[i]["text"], pcs[i], use=("top", "bottom")),
           "replace halves": lambda i: merge(base[i]["text"], pcs[i], replace=True, use=("half1", "half2")),
           "bands guarded": lambda i: base[i]["text"] if turned[i] else merge(base[i]["text"], pcs[i], use=("top", "bottom")),
           "bands clean": lambda i: base[i]["text"] if turned[i] else merge(base[i]["text"], pcs[i], use=("top", "bottom"), clean=True),
           "bands lcnet": lambda i: base[i]["text"] if sure[i] else merge(base[i]["text"], pcs[i], use=("top", "bottom"), clean=True)}
    secs = {"whole": lambda i: base[i]["s"],
            "bands guarded": lambda i: base[i]["s"] + (0 if turned[i] else pcs[i]["top"]["s"] + pcs[i]["bottom"]["s"]),
            "bands clean": lambda i: base[i]["s"] + (0 if turned[i] else pcs[i]["top"]["s"] + pcs[i]["bottom"]["s"]),
            "bands lcnet": lambda i: base[i]["s"] + (0 if sure[i] else pcs[i]["top"]["s"] + pcs[i]["bottom"]["s"]), "insert bands": lambda i: base[i]["s"] + pcs[i]["top"]["s"] + pcs[i]["bottom"]["s"],
            "replace halves": lambda i: base[i]["s"] + pcs[i]["half1"]["s"] + pcs[i]["half2"]["s"]}
    full = lambda i: base[i]["s"] + sum(pcs[i][k]["s"] for k in allp if k in pcs[i])
    need = {"bands guarded": {"top", "bottom"}, "bands clean": {"top", "bottom"}, "bands lcnet": {"top", "bottom"}, "union": set(allp), "insert": set(allp), "replace": set(allp), "insert bands": {"top", "bottom"}, "replace halves": {"half1", "half2"}}
    pol = {p: f for p, f in pol.items() if need.get(p, set()) <= have and (p != "bands lcnet" or ori)}
    texts = {p: {i: f(i) for i in ids} for p, f in pol.items()}
    counts = {p: {m["id"]: (metrics.found(m["entities"], texts[p][m["id"]]), len(m["entities"])) for m in meta} for p in pol}
    refs = {i: corpus.reference(a.corpus, i) for i in ids}
    w = counts["whole"]
    print(f"{a.base} on {a.corpus}, {len(ids)} pages, {sum(n for _, n in w.values())} entities")
    print(f"{'policy':15s} {'recall':>7s} {'diff':>6s} {'95% CI':>16s} {'lost/gained':>11s} {'misses':>7s} {'CER':>6s} {'excess':>7s} {'s/page':>7s}")
    for p in pol:
        c = counts[p]
        r = metrics.recall(c, ids)
        lo, hi = metrics.bootstrap(lambda s: metrics.recall(c, s) - metrics.recall(w, s), ids)
        lost = sum(c[i][0] < w[i][0] for i in ids); gained = sum(c[i][0] > w[i][0] for i in ids)
        miss = sum(n - f for f, n in c.values())
        cer = 100 * statistics.mean(metrics.cer(refs[i], texts[p][i]) for i in ids)
        exc = statistics.median(len(metrics.squash(texts[p][i])) / max(1, len(metrics.squash(refs[i]))) for i in ids)
        s = statistics.mean((secs.get(p) or full)(i) for i in ids)
        print(f"{p:15s} {r:6.2f}% {r - metrics.recall(w, ids):+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {lost:5d}/{gained:<5d} {miss:7d} {cer:5.1f}% {exc:7.3f} {s:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "score"]); ap.add_argument("base"); ap.add_argument("--corpus", required=True)
    ap.add_argument("--only", type=lambda v: v.split(","), default=list(PIECES), help="read only these pieces, e.g. top,bottom")
    ap.add_argument("--exclude", default=None, help="score: a file of page ids to leave out (pages a choice was made on)")
    ap.add_argument("--orient", default=None, help="score: experiments/orient_lcnet.py's answers for the set, for the policy bands lcnet")
    ap.add_argument("--orient-min", type=float, default=0.8, help="score: bands lcnet takes a page for turned at this score or more")
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--mem-gb", type=float, default=None)
    a = ap.parse_args()
    if corpus.role(a.corpus) == "test" and not corpus.ledger_has(f"{a.base}_pieces"):
        sys.exit(f"'{a.corpus}' is a test set: write the question and the tag `{a.base}_pieces` in docs/LEDGER.md first.")
    {"read": cmd_read, "score": cmd_score}[a.cmd](a)
