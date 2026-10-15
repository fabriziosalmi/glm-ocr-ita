"""work/gen_venv/bin/python generators/make_gu.py [--issues-per-year 6] [--pages 4] : real Italian printed pages with exact ground truth,
from the Gazzetta Ufficiale della Repubblica Italiana, Serie Generale (www.gazzettaufficiale.it; docs/EXTERNAL.md).
-> external/gu_src/ (the issues, as downloaded), external/gu_dev/ and external/gu_test/ ({id}.png, {id}.txt, meta.json).

Why. Every Italian page measured so far was either synthetic or a form (XFUND). The Gazzetta is real Italian typesetting, two
columns, small print, decrees, notices, tables, written by many offices, and its PDFs are born digital: the text layer is the
ground truth, exactly. Official acts are excluded from copyright in Italy (law 633/1941, art. 5). Nothing leaves external/.
Sampling. Years 2019 to 2025; from each year --issues-per-year issues drawn with a fixed seed from the year's archive, then
--pages pages per issue (never the cover). Issues alternate between dev and test in the drawn order, so a page set never shares
an issue with the other: the same decree cannot sit on both sides.
Ground truth. The text layer in its own order (column by column, as printed), with words hyphenated at a line end joined again
(pdfium marks those hyphens with U+FFFE). Running headers are kept: they are on the page.
Entities. Taken by pattern from the ground truth, running header and page number left out (entities() says why): dates written out ("6 settembre 2005") or in figures ("1-9-2023"), act numbers
("n. 206"), amounts with a decimal comma ("1.234,56"). Whitespace is ignored when they are matched, so a line break inside one
does not matter. Fixed before any model read a page.
Images: rendered from the PDF with the long side at 1754 px (A4 at 150 dpi), as every other page set.

--train: training pages for the fine-tuning op3 (2026-10-15) -> external/gu_train. Other issues (seed + 1), never one of
gu_dev or gu_test; from each, the pages with an amount of a million or more ("29.467.745,94", which v1.1 sometimes writes
"29.467,745,94") or a row of dot leaders, else none. Level 1 (born digital)."""
import argparse, json, pathlib, random, re, time, urllib.request
import pypdfium2 as pdfium

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC, BASE = ROOT / "external/gu_src", "https://www.gazzettaufficiale.it"
UA = {"User-Agent": "Mozilla/5.0 (glm-ocr-ita research; page set for OCR evaluation)"}
MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
PATTERNS = {"date": rf"\b\d{{1,2}}°?\s+(?:{MONTHS})\s+\d{{4}}\b", "date_num": r"\b\d{1,2}-\d{1,2}-\d{4}\b",
            "act_number": r"\bn\.\s?\d{1,5}(?:/\d{2,4})?\b", "amount": r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b"}


def get(url):
    time.sleep(1.0)   # one request a second: a public service
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def issues(year):
    """(date, number) of every Serie Generale issue of a year, from the site's archive page"""
    cache = SRC / f"index_{year}.html"
    if not cache.exists():
        cache.write_bytes(get(f"{BASE}/ricercaArchivioCompleto/serie_generale/{year}"))
    found = re.findall(r"dataPubblicazioneGazzetta=(\d{4}-\d{2}-\d{2})&(?:amp;)?numeroGazzetta=(\d+)", cache.read_text(errors="replace"))
    return sorted(set(found), key=lambda x: (x[0], int(x[1])))


def issue_pdf(date, number):
    p = SRC / f"gu_{date}_{int(number):03d}.pdf"
    if not p.exists():
        y, m, d = date.split("-")
        data = get(f"{BASE}/eli/gu/{y}/{m}/{d}/{number}/sg/pdf")
        assert data[:4] == b"%PDF", f"not a PDF: {date} n. {number}"
        p.write_bytes(data)
    return p


def page_text(page):
    t = page.get_textpage().get_text_range()
    t = re.sub(r"￾\s*\r?\n", "", t).replace("￾", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in t.split("\n") if line.strip())


def entities(text):
    """from the body only: the page number ("— 5 —") and the running header (issue date, title, number) are left out. Both
    models skip running headers, as most OCR does, and a header entity would measure that habit, not the reading of the page.
    Decided on 2026-10-14 after the first dev reads, before any read of gu_test."""
    body = [l for l in text.split("\n") if not re.fullmatch(r"—\s*\d+\s*—", l) and "GAZZETTA UFFICIALE DELLA REPUBBLICA ITALIANA" not in l]
    flat = re.sub(r"\s+", " ", "\n".join(body)); out = []
    for typ, pat in PATTERNS.items():
        out += [{"type": typ, "text": m.group(0)} for m in re.finditer(pat, flat, flags=re.I)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs="+", default=list(range(2019, 2026)))
    ap.add_argument("--issues-per-year", type=int, default=6); ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261014); ap.add_argument("--train", type=int, default=0, help="pages for gu_train (0: build dev and test)")
    a = ap.parse_args()
    if a.train:
        taken = {m["issue"] for r in ("dev", "test") for m in json.load(open(ROOT / f"external/gu_{r}/meta.json"))}
        rng = random.Random(a.seed + 1); out = ROOT / "external/gu_train"; out.mkdir(exist_ok=True); meta = []
        want = re.compile(r"\b\d{1,3}\.\d{3}\.\d{3},\d{2}\b|(?:\.\s?){12,}")
        pool = [(y, *iss) for y in a.years for iss in issues(y) if f"{iss[0]} n. {iss[1]}" not in taken]
        for year, date, number in rng.sample(pool, len(pool)):
            if len(meta) >= a.train: break
            pdf = pdfium.PdfDocument(str(issue_pdf(date, number)))
            good = [i for i in range(1, len(pdf)) if want.search(page_text(pdf[i]))]
            for i in rng.sample(good, min(2, len(good))):
                page = pdf[i]; pid = f"gu_{date}_{int(number):03d}_p{i + 1:02d}"; text = page_text(page)
                img = page.render(scale=1754 / max(page.get_size())).to_pil().convert("RGB")
                img.save(out / f"{pid}.png"); (out / f"{pid}.txt").write_text(text, encoding="utf-8")
                meta.append({"id": pid, "gt": "pdf_text_layer", "level": 1, "tpl": "gazzetta", "lang": "it", "rot90": False,
                             "size": list(img.size), "issue": f"{date} n. {number}", "page": i + 1, "entities": entities(text)})
            print(date, number, len(good), "pages wanted")
        json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(out, len(meta), "pages,", sum(len(m["entities"]) for m in meta), "entities"); raise SystemExit
    SRC.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed)
    drawn = [(y, *iss) for y in a.years for iss in rng.sample(issues(y), a.issues_per_year)]
    sets = {"dev": [], "test": []}
    for k, (year, date, number) in enumerate(drawn):
        role = ("dev", "test")[k % 2]; out = ROOT / f"external/gu_{role}"; out.mkdir(exist_ok=True)
        pdf = pdfium.PdfDocument(str(issue_pdf(date, number)))
        candidates = [i for i in range(1, len(pdf)) if len(page_text(pdf[i])) > 400]   # no cover, no near-empty page
        for i in sorted(rng.sample(candidates, min(a.pages, len(candidates)))):
            page = pdf[i]; pid = f"gu_{date}_{int(number):03d}_p{i + 1:02d}"
            text = page_text(page)
            img = page.render(scale=1754 / max(page.get_size())).to_pil().convert("RGB")
            img.save(out / f"{pid}.png"); (out / f"{pid}.txt").write_text(text, encoding="utf-8")
            sets[role].append({"id": pid, "gt": "pdf_text_layer", "level": 1, "tpl": "gazzetta", "lang": "it", "rot90": False,
                               "size": list(img.size), "issue": f"{date} n. {number}", "page": i + 1, "entities": entities(text)})
        print(role, date, number, len(pdf), "pages")
    for role, meta in sets.items():
        json.dump(meta, open(ROOT / f"external/gu_{role}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n = lambda t: sum(e["type"] == t for m in meta for e in m["entities"])
        print(role, len(meta), "pages,", sum(len(m["entities"]) for m in meta), "entities:", {t: n(t) for t in PATTERNS})
