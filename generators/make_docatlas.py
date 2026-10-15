"""python generators/make_docatlas.py [--docs 40] : real Italian documents with exact ground truth, from DocAtlas
(ahmedheakl/docatlas_instruct, CC BY 4.0; docs/EXTERNAL.md), as downloaded under ~/Documents/git/datasets/huggingface/data/
by download_docatlas_it.py (clean renders) and scan_fx.py (the same pages made to look scanned).
-> external/docatlas_it_test/ (clean) and external/docatlas_it_scan_test/ (the same pages, scanned): {id}.png, {id}.txt, meta.json,
and external/docatlas_test_docs.json, the documents reserved for testing: a training set drawn from DocAtlas must leave them out.

DocAtlas pages are real documents found on the web (tenders, forms, notices, reports), rendered from their DOCX, so the text of
every block is known exactly. --docs documents are drawn with a fixed seed, one page from each, so that no document weighs more
than another; the clean and the scanned set hold the same pages, which measures what the scan alone costs.
Ground truth: the dataset's markdown with its HTML table tags removed (one cell per line) and image placeholders dropped.
Entities, by pattern on the ground truth, as for the Gazzetta (generators/make_gu.py) plus the codes of Italian paperwork:
dates, act numbers, amounts with a decimal comma, VAT numbers, tax codes, IBAN, CIG, e-mail and PEC addresses; and every
distinct word of 8 letters or more (entities() says why).
Images: long side 1754 px (A4 at 150 dpi), as every other page set. Fixed before any model read a page.

--train: training pages for the fine-tuning op3 (2026-10-15), from the documents NOT reserved for test, chosen for what v1.1
reads badly: blanks drawn as rows of dots or underscores (it writes the row and stops), tick boxes (it repeats empty boxes), and
amounts of a million or more. Markdown escapes are removed from their ground truth (plain(unescape=True)). At most one page per
document; every other page taken from the scanned version (level 2), the
others clean (level 1). -> external/docatlas_train."""
import argparse, json, os, pathlib, random, re
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent; os.chdir(ROOT)
SRC = pathlib.Path.home() / "Documents/git/datasets/huggingface/data"
MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
PATTERNS = {"date": rf"\b\d{{1,2}}°?\s+(?:{MONTHS})\s+\d{{4}}\b|\b\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{4}}\b",
            "act_number": r"\bn\.\s?\d{1,5}(?:/\d{2,4})?\b", "amount": r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b",
            "vat": r"\b\d{11}\b", "tax_code": r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
            "iban": r"\bIT\d{2}[A-Z]\d{10}[A-Z0-9]{12}\b", "cig": r"\bCIG:?\s*[A-Z0-9]{10}\b", "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"}


MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")


def plain(markdown, unescape=False):
    """unescape: markdown escapes ("\\_" for every underscore of a blank) back to the characters printed on the page. Off for the
    test sets, whose ground truth was fixed and read before this was found (2026-10-15); their entities hold no escape."""
    t = markdown.replace("<!-- image -->", "")
    if unescape:
        t = MD_ESCAPE.sub(r"\1", t)
    t = re.sub(r"</t[dh]>\s*", "\n", t); t = re.sub(r"<[^>]+>", "", t)
    return "\n".join(l.strip() for l in t.split("\n") if l.strip())


def entities(text):
    """the patterns below, plus every distinct word of 8 letters or more ("word"): DocAtlas is mostly prose, where the patterns
    find few values (118 on 40 pages, none on 18 of them), and long words measure how much of the text comes back as written,
    accents included; short words are left out because they match anywhere"""
    flat = re.sub(r"\s+", " ", text)
    out = [{"type": "word", "text": w} for w in dict.fromkeys(re.findall(r"[^\W\d_]{8,}", flat))]
    for typ, pat in PATTERNS.items():
        out += [{"type": typ, "text": m.group(0)} for m in re.finditer(pat, flat, flags=re.I if typ in ("date", "act_number", "email") else 0)]
    return out


NEEDS = {"blank": r"\.{10,}|(?:\. ){8,}|_{10,}|…{4,}", "box": r"[☐☒☑□]", "million": r"\b\d{1,3}\.\d{3}\.\d{3},\d{2}\b"}


def save(img, out, i, text, extra):
    k = 1754 / max(img.size)
    if k < 1: img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    img.save(out / f"{i}.png"); (out / f"{i}.txt").write_text(text, encoding="utf-8")
    return {"id": i, "gt": "docx_render", "tpl": "docatlas", "lang": "it", "rot90": False, "size": list(img.size), **extra, "entities": entities(text)}


def train_set(clean, scan, rng, caps={"blank": 45, "box": 20, "million": 11}):
    test = set(json.load(open("external/docatlas_test_docs.json"))); used = set(); meta = []
    out = pathlib.Path("external/docatlas_train"); out.mkdir(exist_ok=True)
    for need, cap in caps.items():
        pool = [i for i in sorted(clean) if i in scan and i.rsplit("_p", 1)[0] not in test | used and re.search(NEEDS[need], clean[i]["markdown"])]
        rng.shuffle(pool); taken = 0
        for i in pool:
            d = i.rsplit("_p", 1)[0]
            if d in used or taken == cap: continue
            taken += 1
            used.add(d); scanned = len(meta) % 2 == 1; r = (scan if scanned else clean)[i]
            img = Image.open(SRC / ("docatlas_it_scan" if scanned else "docatlas_it") / r["image"]).convert("RGB")
            meta.append(save(img, out, i, plain(clean[i]["markdown"], unescape=True), {"level": 2 if scanned else 1, "need": need, "document": d}))
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out, len(meta), "pages", {n: sum(m["need"] == n for m in meta) for n in caps}, "scanned", sum(m["level"] == 2 for m in meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", type=int, default=40); ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--train", action="store_true"); a = ap.parse_args(); rng = random.Random(a.seed)
    clean = {r["id"]: r for r in map(json.loads, open(SRC / "docatlas_it/labels.jsonl"))}
    scan = {r["id"]: r for r in map(json.loads, open(SRC / "docatlas_it_scan/labels.jsonl"))}
    if a.train:
        train_set(clean, scan, random.Random(a.seed + 2)); raise SystemExit
    by_doc = {}
    for i in sorted(clean):
        if i in scan and len(plain(clean[i]["markdown"])) > 300: by_doc.setdefault(i.rsplit("_p", 1)[0], []).append(i)
    docs = rng.sample(sorted(by_doc), a.docs)
    for name, rows, sub in (("docatlas_it_test", clean, "docatlas_it"), ("docatlas_it_scan_test", scan, "docatlas_it_scan")):
        out = pathlib.Path("external") / name; out.mkdir(exist_ok=True); meta = []
        pick = random.Random(a.seed + 1)   # the same page of each document in both sets
        for d in docs:
            i = pick.choice(by_doc[d]); r = rows[i]; text = plain(clean[i]["markdown"])
            img = Image.open(SRC / sub / r["image"]).convert("RGB"); k = 1754 / max(img.size)
            if k < 1: img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
            img.save(out / f"{i}.png"); (out / f"{i}.txt").write_text(text, encoding="utf-8")
            meta.append({"id": i, "gt": "docx_render", "level": 1, "tpl": "docatlas", "lang": "it", "rot90": False, "size": list(img.size),
                         "document": d, **({"fx": r.get("fx", {}).get("profile")} if sub.endswith("scan") else {}), "entities": entities(text)})
        json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = lambda t: sum(e["type"] == t for m in meta for e in m["entities"])
        print(name, len(meta), "pages,", sum(len(m["entities"]) for m in meta), "entities", {t: n(t) for t in ["word", *PATTERNS]})
    json.dump(sorted(docs), open("external/docatlas_test_docs.json", "w"), indent=1)
