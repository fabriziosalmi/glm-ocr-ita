"""Static audit and freeze of the commissioned page sets (gemini, spark, chatgpt). No model, no GPU.

    .venv/bin/python experiments/dataset_audit.py manifest   # SHA-256 of meta.json and every .txt; PNGs by os.stat only
    .venv/bin/python experiments/dataset_audit.py text       # per-set analysis of meta.json and .txt -> cache/*.pkl
    .venv/bin/python experiments/dataset_audit.py stats      # cache -> stats.json
    .venv/bin/python experiments/dataset_audit.py report     # stats.json -> report.html (+ PDF with --pdf PATH)
    .venv/bin/python experiments/dataset_audit.py all --pdf ~/Desktop/Dataset-audit.pdf

Everything is written under work/dataset_audit/. The datasets are only read. The steps are independent: an `images`
step (PNG hashes, pixel statistics) can be added later and write its own file without redoing `text`.
Images are NOT opened by this script: the manifest covers meta.json and .txt only.
"""
import argparse
import bisect
import collections
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import re
import statistics
import subprocess
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from glmocr import corpus, metrics  # noqa: E402

OUT = corpus.WORK / "dataset_audit"
CACHE = OUT / "cache"
ROOT = corpus.DATASETS
RENDERERS = {"gemini": "abcdefghijklmnopqrstuvwxyz", "spark": "abcdefghijk", "chatgpt": "abcdefghijklmnopqrstuvwxyz"}
SETS = [f"{r}/itdoc_{c}" for r, cs in RENDERERS.items() for c in cs]
WORKERS = 4
MAX_TOKENS = metrics.MAX_TOKENS
CANON = ["amount", "date", "iban", "vat", "cf", "cig", "cup", "contract", "pec", "percent", "hand"]
META_KEYS = ["id", "gt", "level", "tpl", "lang", "font", "rot90", "seed", "small_print_px", "degradation", "overlays", "entities"]
DEG_KEYS = ["rotation_deg", "blur", "noise_sigma", "jpeg_quality"]


def role_of(name):
    return corpus.registry().get(name, {}).get("role")


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest

def manifest_set(name):
    d = ROOT / name
    listed = {m["id"] for m in json.loads((d / "meta.json").read_text(encoding="utf-8"))}
    rows, png_n, png_bytes, other, stray = [], 0, 0, [], []
    for e in sorted(os.scandir(d), key=lambda e: e.name):
        if not e.is_file():
            other.append(e.name + "/")
            continue
        stem, ext = os.path.splitext(e.name)
        if ext in (".png", ".txt") and stem not in listed:
            stray.append(e.name)
        if ext == ".png":
            png_n += 1
            png_bytes += e.stat().st_size
        elif ext == ".txt" or e.name == "meta.json":
            rows.append((sha_file(e.path), e.name, e.stat().st_size))
        else:
            other.append(e.name)
    lines = sorted(f"{h}  {n}" for h, n, _ in rows)
    return name, {
        "files_hashed": len(rows), "bytes_hashed": sum(s for *_, s in rows),
        "txt_files": sum(n.endswith(".txt") for _, n, _ in rows),
        "digest": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        "meta_json_sha256": next(h for h, n, _ in rows if n == "meta.json"),
        "pages_in_meta": len(listed),
        "png_files_NOT_hashed": png_n, "png_bytes_NOT_hashed": png_bytes,
        "files_not_listed_in_meta": stray, "other_entries": other,
    }, [f"{h}  {name}/{n}" for h, n, _ in rows]


def step_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(WORKERS) as pool:
        res = pool.map(manifest_set, SETS)
    lines = sorted((l for _, _, ls in res for l in ls), key=lambda l: l.split("  ", 1)[1])
    (OUT / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sets = {n: d for n, d, _ in res}
    tot = lambda k: sum(d[k] for d in sets.values())
    doc = {
        "what": "Freeze manifest of the commissioned page sets. TEXT AND METADATA ONLY: meta.json and every .txt of each set.",
        "images": "NOT COVERED. No .png was read or hashed. png_files_NOT_hashed and png_bytes_NOT_hashed come from os.stat only.",
        "root": str(ROOT), "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "how_to_verify": "cd <root> && shasum -a 256 -c manifest.sha256 ; per-set digest = SHA-256 of the sorted lines "
                         "'<sha256>  <filename>' (two spaces, file name without folder) joined by \\n, with a final \\n.",
        "manifest_sha256": sha_file(OUT / "manifest.sha256"),
        "totals": {"sets": len(sets), "files_hashed": tot("files_hashed"), "bytes_hashed": tot("bytes_hashed"),
                   "png_files_NOT_hashed": tot("png_files_NOT_hashed"), "png_bytes_NOT_hashed": tot("png_bytes_NOT_hashed")},
        "sets": sets,
    }
    (OUT / "digests.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print("manifest:", doc["totals"])


# ---------------------------------------------------------------- validity of codes

_ODD = dict(zip("0123456789", [1, 0, 5, 7, 9, 13, 15, 17, 19, 21]))
_ODD.update(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]))
_EVEN = {c: i for i, c in enumerate("0123456789")}
_EVEN.update({c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})


def _ctl(s):
    """Control letter of the codice fiscale, and of the CIN of an Italian BBAN: same tables."""
    return chr(65 + sum((_ODD if i % 2 == 0 else _EVEN)[c] for i, c in enumerate(s)) % 26)


def iban_check(text):
    """shape: IT + 25 letters or digits. format: the Italian layout, IT, 2 digits, the CIN letter, 22 more.
    mod97 is computed on the shape alone, so an IBAN with no CIN letter can still pass it."""
    s = re.sub(r"\s+", "", text).upper()
    shape = bool(re.fullmatch(r"IT\d{2}[0-9A-Z]{23}", s))
    fmt = bool(re.fullmatch(r"IT\d{2}[A-Z]\d{10}[0-9A-Z]{12}", s))
    mod97 = shape and int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1
    return {
        "shape": shape,
        "format": fmt,
        "mod97": mod97,
        "cin": fmt and _ctl(s[5:]) == s[4],
        "all_valid": fmt and mod97 and _ctl(s[5:]) == s[4],
        "grouped4": bool(re.fullmatch(r"(\S{4} ){6}\S{3}", text.strip())),
    }


def luhn11(d):
    s = sum(int(c) for c in d[0:10:2]) + sum((2 * int(c) - 9) if int(c) > 4 else 2 * int(c) for c in d[1:10:2])
    return (10 - s % 10) % 10 == int(d[10])


def vat_check(text):
    s = re.sub(r"[\s.]", "", text).upper().removeprefix("IT")
    fmt = bool(re.fullmatch(r"\d{11}", s))
    return {"format": fmt, "check": fmt and luhn11(s)}


CF_SHAPE = re.compile(r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]")


def cf_check(text):
    """shape: 6 letters, 2 digits, letter, 2 digits, letter, 3 digits, letter. check: the control letter, computed on
    the shape alone. format: shape, a month letter that exists and a day between 01-31 or 41-71. all_valid: both."""
    s = re.sub(r"\s+", "", text).upper()
    if re.fullmatch(r"\d{11}", s):  # the codice fiscale of a company is its partita IVA
        ok = luhn11(s)
        return {"shape": True, "format": True, "check": ok, "numeric": True, "month_ok": True, "day_ok": True, "all_valid": ok}
    shape = bool(CF_SHAPE.fullmatch(s))
    day = int(s[9:11]) if shape else 0
    month = shape and s[8] in "ABCDEHLMPRST"
    day_ok = shape and (1 <= day <= 31 or 41 <= day <= 71)
    check = shape and _ctl(s[:15]) == s[15]
    return {"shape": shape, "format": shape and month and day_ok, "check": check, "numeric": False, "month_ok": month,
            "day_ok": day_ok, "all_valid": check and month and day_ok}


IT_MONTHS = "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split()
EN_MONTHS = "january february march april may june july august september october november december".split()


def _valid(y, m, d):
    try:
        datetime.date(int(y), int(m), int(d))
        return True
    except ValueError:
        return False


def date_check(text):
    """(format family, calendar-valid). Families: it_num dd/mm/yyyy dd.mm.yyyy dd-mm-yyyy; iso; it_words; en_words; other."""
    t = text.strip().lower()
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        return "iso", _valid(*m.groups())
    if m := re.search(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})\b", t):
        d, _, mo, y = m.groups()
        if _valid(y, mo, d):
            return "it_num", True
        return ("us_num", True) if _valid(y, d, mo) else ("it_num", False)
    if m := re.search(r"\b(\d{1,2})°?\s+([a-zà-ÿ]+)\s+(\d{4})\b", t):
        d, w, y = m.groups()
        if w in IT_MONTHS:
            return "it_words", _valid(y, IT_MONTHS.index(w) + 1, d)
        if w in EN_MONTHS:
            return "en_words", _valid(y, EN_MONTHS.index(w) + 1, d)
    if m := re.search(r"\b([a-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", t):
        w, d, y = m.groups()
        if w in EN_MONTHS:
            return "en_words", _valid(y, EN_MONTHS.index(w) + 1, d)
    return "other", False


def number_notation(text):
    """it: 1.234,56  en: 1,234.56  plain: no separator  other: anything else. Second value: thousands groups."""
    m = re.search(r"\d[\d.,]*\d|\d", text)
    if not m:
        return "other", 0
    n = m.group(0)
    if re.fullmatch(r"\d+", n):
        return "plain", 0
    if re.fullmatch(r"\d{1,3}(\.\d{3})*(,\d{1,2})?", n) or re.fullmatch(r"\d+,\d{1,2}", n):
        return "it", n.count(".")
    if re.fullmatch(r"\d{1,3}(,\d{3})*(\.\d{1,2})?", n) or re.fullmatch(r"\d+\.\d{1,2}", n):
        return "en", n.count(",")
    return "other", 0


def tail_kind(t):
    """A type outside the eleven of the brief whose name says it holds a checkable code."""
    if "iban" in t:
        return "iban"
    if re.search(r"(^|_)(vat|piva|partita_iva)(_|$)", t):
        return "vat"
    if re.search(r"fiscal_code|codice_fiscale|(^|_)cf(_|$)", t):
        return "cf"
    return None


# ---------------------------------------------------------------- language, unicode

IT_STOP = set("il lo la le gli di del dello della dei degli delle che è e ed per con non una un uno al allo alla ai agli alle nel nello nella nei nelle sono da dal dalla dai dalle sul sulla sui sulle si o come anche tra fra presso entro ogni essere sia salvo qualora ovvero nonché più quale quali questo questa suo sua loro deve devono viene vengono è stato stata ha hanno".split())
EN_STOP = set("the of and to is for with by this that be shall at on as or are from not any which has have will under within an its been was were all may must each such other their these those if upon into than shall it hereby between".split())
IT_LABEL = set("codice fiscale partita iva data importo totale numero sede via firma pagamento fattura scadenza cliente fornitore indirizzo targa descrizione quantità prezzo oggetto comune provincia nato residente telefono cognome nome luogo nascita imponibile aliquota imposta bollo ragione sociale intestatario causale banca conto ufficio protocollo allegato articolo contratto ordine consegna spedizione destinatario mittente telaio veicolo polizza premio rata saldo dichiarazione".split())
EN_LABEL = set("code tax date amount total number office street signature payment invoice due customer supplier address plate description quantity price subject municipality province born resident phone surname name place birth taxable rate stamp company holder reason bank account protocol annex article contract order delivery shipment consignee sender chassis vehicle policy premium instalment balance statement".split())
WORD = re.compile(r"[a-zà-ÿ]+")
NO_ACCENT = set("perche poiche piu cioe gia societa qualita attivita modalita responsabilita validita facolta sara puo cosi finche benche affinche nonche citta unita quantita proprieta autorita eta liberta necessita possibilita capacita conformita".split())
YES_ACCENT = set("perché poiché più cioè già società qualità attività modalità responsabilità validità facoltà sarà può così finché benché affinché nonché città unità quantità proprietà autorità età libertà necessità possibilità capacità conformità".split())
CHAR_CLASS = {
    "—": "em_dash", "–": "en_dash", "−": "minus_sign", "‐": "hyphen_u2010", "‑": "nb_hyphen",
    "‘": "curly_single", "’": "curly_single", "“": "curly_double", "”": "curly_double",
    "«": "guillemet", "»": "guillemet", " ": "nbsp", " ": "nbsp", " ": "nbsp", " ": "nbsp",
    "​": "zero_width", "‌": "zero_width", "‍": "zero_width", "﻿": "zero_width", "­": "soft_hyphen",
    "…": "ellipsis", "•": "bullet", "·": "bullet", "°": "degree", "º": "ordinal", "ª": "ordinal",
    "№": "numero_sign", "§": "section", "€": "euro", "£": "pound", "×": "times", "\t": "tab", "\r": "cr",
}
ESCAPES = {"backslash_n": re.compile(r"\\n"), "backslash_t": re.compile(r"\\t"), "backslash_u": re.compile(r"\\u[0-9a-fA-F]{4}"),
           "backslash_x": re.compile(r"\\x[0-9a-fA-F]{2}"), "html_entity": re.compile(r"&(amp|nbsp|lt|gt|quot|#\d+);"),
           "html_tag": re.compile(r"</?(br|b|i|p|div|span|table|td|tr)\b[^>]*>", re.I), "asterisk_run": re.compile(r"\*\*"),
           "markdown_heading": re.compile(r"(?m)^#{1,4} ")}
APOS_ACCENT = re.compile(r"\b[a-zA-Z]+[aeiou]['’](?=\s|$|[.,;:)])")
COMPANY = re.compile(r"((?:[A-ZÀ-Ý][\w&'’.-]*\s+){1,4}?)(S\.r\.l\.s\.|S\.r\.l\.|S\.p\.A\.|S\.n\.c\.|S\.a\.s\.|S\.c\.a\.r\.l\.|S\.c\.p\.A\.|Ltd\.?|LLC|Inc\.|GmbH)")
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
TOKEN = re.compile(r"[^\W\d_]+")
IBAN_IN_TEXT = re.compile(r"(?<![a-z])it\d{2}[a-z0-9]\d{10}[0-9a-z]{12}(?![0-9])")


def pattern_of(s):
    """The sentence without its slots: only the words written in lower case (the first word is lowered first)."""
    toks = TOKEN.findall(s)
    if toks:
        toks[0] = toks[0].lower()
    return " ".join(t for t in toks if t.islower() and len(t) > 1)


def line_start(l):
    return " ".join(re.sub(r"\d", "0", l.lower()).split()[:3])


# ---------------------------------------------------------------- per-set text analysis

def analyse_set(name):
    d = ROOT / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    letter = name[-1]
    pages, texts, raw_lines = [], [], []
    C = collections.Counter
    sent_pages, sent_chars, pat_pages, pat_chars = C(), C(), C(), C()
    sent_example = {}
    chars, companies, apos_words = C(), C(), C()
    ent_strings = C()                      # (type, squashed) -> occurrences
    ent = collections.defaultdict(C)       # type -> counters
    tpl_starts = collections.defaultdict(C)  # tpl -> line start -> pages (Italian pages only)
    tpl_it_pages = C()
    ent_keys, deg_keys, meta_keys = C(), C(), C()
    bbox_values = C()
    pos_dec = collections.defaultdict(C)
    for idx, m in enumerate(meta):
        raw = (d / f"{m['id']}.txt").read_bytes()
        rawtext = raw.decode("utf-8")
        text = rawtext.replace("\\n", "\n") if corpus.escaped_breaks(rawtext) else rawtext
        lines = text.split("\n")
        nonempty = [l for l in lines if l.strip()]
        sq_lines = [metrics.squash(l) for l in nonempty]
        sq = "".join(sq_lines)
        offs, o = [], 0
        for s in sq_lines:
            offs.append(o)
            o += len(s)
        low = text.lower()
        words = WORD.findall(low)
        it_s, en_s = sum(w in IT_STOP for w in words), sum(w in EN_STOP for w in words)
        it_l, en_l = sum(w in IT_LABEL for w in words), sum(w in EN_LABEL for w in words)
        cls = C()
        for ch in text:
            if ord(ch) > 126 or (ord(ch) < 32 and ch != "\n"):
                chars[ch] += 1
                if ch in CHAR_CLASS:
                    cls[CHAR_CLASS[ch]] += 1
        esc = {k: len(r.findall(rawtext)) for k, r in ESCAPES.items()}
        for w in APOS_ACCENT.findall(text):
            apos_words[w.lower()] += 1
        psent, ppat = {}, {}
        for l in nonempty:
            for s in SENT_SPLIT.split(l):
                k = metrics.squash(s)
                if len(k) >= 30:
                    hk = h64(k)
                    psent[hk] = len(k)
                    sent_example.setdefault(hk, s.strip()[:200])
                    p = pattern_of(s)
                    if p.count(" ") >= 4:
                        ppat[h64(p)] = len(k)
        for hk, n in psent.items():
            sent_pages[hk] += 1
            sent_chars[hk] += n
        for hk, n in ppat.items():
            pat_pages[hk] += 1
            pat_chars[hk] += n
        for mm in COMPANY.finditer(text):
            companies[" ".join((mm.group(1) + mm.group(2)).split())] += 1
        if m.get("lang") == "it":
            tpl_it_pages[m.get("tpl")] += 1
            for st in {line_start(l) for l in nonempty}:
                tpl_starts[m.get("tpl")][st] += 1
        meta_keys.update(m.keys())
        deg = m.get("degradation") or {}
        deg_keys.update(deg.keys())
        idm = re.fullmatch(rf"itdoc_{letter}_(\d{{4}})_(.+)_L(\d)_(it|en)", m["id"])
        ida = re.fullmatch(rf"itdoc_{letter}_(\d{{4}})_L(\d)_(it|en)_(.+)", m["id"])
        id_alt = bool(ida) and int(ida.group(1)) == idx and ida.group(4) == m.get("tpl") and int(ida.group(2)) == m.get("level") and ida.group(3) == m.get("lang")
        ibans_text = set(IBAN_IN_TEXT.findall(sq))
        pages.append({
            "id": m["id"], "idx": idx, "level": m.get("level"), "lang": m.get("lang"), "tpl": m.get("tpl"), "font": m.get("font"),
            "rot90": bool(m.get("rot90")), "seed": m.get("seed"), "small_print_px": m.get("small_print_px"),
            "min_cap_height": m.get("min_cap_height"),
            "deg": {k: deg.get(k) for k in DEG_KEYS}, "deg_extra": sorted(set(deg) - set(DEG_KEYS)),
            "n_overlays": len(m.get("overlays") or []), "missing_keys": [k for k in META_KEYS if k not in m],
            "id_ok": bool(idm) and int(idm.group(1)) == idx and idm.group(2) == m.get("tpl") and int(idm.group(3)) == m.get("level") and idm.group(4) == m.get("lang"),
            "id_other_order": id_alt,
            "txt_sha": hashlib.sha256(raw).hexdigest(), "digit_hash": h64(re.sub(r"\d", "0", sq)), "pattern_hash": h64(pattern_of(text)),
            "bytes": len(raw), "chars": len(text), "sq_chars": len(sq), "words": len(text.split()), "lines": len(nonempty),
            "empty_lines": len(lines) - len(nonempty) - (1 if text.endswith("\n") else 0), "max_line": max(map(len, lines)),
            "ends_newline": text.endswith("\n"), "edge_space_lines": sum(l != l.strip() for l in nonempty),
            "double_space": text.count("  "), "pipe_lines": sum(" | " in l for l in nonempty),
            "nfc": text == unicodedata.normalize("NFC", text), "nfkc": text == unicodedata.normalize("NFKC", text),
            "escaped_breaks": corpus.escaped_breaks(rawtext), "esc": {k: v for k, v in esc.items() if v}, "cls": dict(cls),
            "apos_accent": len(APOS_ACCENT.findall(text)), "no_accent": sum(w in NO_ACCENT for w in words), "yes_accent": sum(w in YES_ACCENT for w in words),
            "it_stop": it_s, "en_stop": en_s, "it_label": it_l, "en_label": en_l,
            "n_sent": len(psent), "n_ent": len(m["entities"]), "ibans_in_text": len(ibans_text),
        })
        texts.append((sq, sq_lines, offs, ibans_text))
        raw_lines.append(nonempty)
    n = len(meta)
    for idx, m in enumerate(meta):
        sq, sq_lines, offs, ibans_text = texts[idx]
        other_sq = texts[(idx + 1) % n][0]
        es = [(e.get("type") or "?", metrics.squash(e["text"]), e) for e in m["entities"]]
        seen = C(s for _, s, _ in es)
        distinct = set(seen)
        pg = pages[idx]
        amb = miss = short = 0
        listed_ibans = set()
        nl = len(sq_lines)
        for t, s, e in es:
            ent_keys.update(e.keys())
            if "bbox" in e:
                bbox_values[tuple(e["bbox"])] += 1
            tg = t if t in CANON else "other"
            c = ent[tg]
            c["n"] += 1
            c["small_print_flag"] += bool(e.get("small_print"))
            ent_strings[(t, s)] += 1
            first = sq.find(s) if s else -1
            if first < 0:
                c["missing"] += 1
                miss += 1
                continue
            k, start = 0, first
            while start >= 0:
                k += 1
                start = sq.find(s, start + 1)
            is_sub = any(s != o and s in o for o in distinct)
            c["short"] += len(s) < 5
            short += len(s) < 5
            c["multi"] += k > 1
            c["sub"] += is_sub
            c["dup_listed"] += seen[s] > 1
            c["ambiguous"] += k > 1 or is_sub
            amb += k > 1 or is_sub
            c["in_next_page"] += s in other_sq
            li = bisect.bisect_right(offs, first) - 1
            c["first3"] += li < 3
            c["last3"] += li >= nl - 3 and li >= 3
            pos_dec[tg][min(9, int(10 * li / max(nl, 1)))] += 1
            c["nonascii"] += any(ord(ch) > 126 for ch in e["text"])
            for ch in set(e["text"]):
                if ch in CHAR_CLASS:
                    c["has_" + CHAR_CLASS[ch]] += 1
            kind = tg if tg != "other" else tail_kind(t)
            pre = "" if tg != "other" else "tail_"
            lang = m.get("lang")
            if kind == "iban":
                r = iban_check(e["text"])
                listed_ibans.add(re.sub(r"\s+", "", e["text"]).lower())
                for kk, v in r.items():
                    c[f"{pre}iban_{kk}"] += v
                c[pre + "iban_n"] += 1
            elif kind == "vat":
                r = vat_check(e["text"])
                c[pre + "vat_format"] += r["format"]
                c[pre + "vat_check"] += r["check"]
                c[pre + "vat_n"] += 1
            elif kind == "cf":
                r = cf_check(e["text"])
                for kk, v in r.items():
                    c[f"{pre}cf_{kk}"] += v
                c[pre + "cf_n"] += 1
            elif tg == "cig":
                c["cig_format"] += bool(re.fullmatch(r"[0-9A-Z]{10}", e["text"].strip()))
                c["cig_real_shape"] += bool(re.fullmatch(r"[0-9A-Z]{7}[0-9A-F]{3}", e["text"].strip()))
            elif tg == "cup":
                c["cup_format"] += bool(re.fullmatch(r"[A-Z]\d{2}[A-Z]\d{11}", e["text"].strip()))
            elif tg == "pec":
                ok = bool(re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", e["text"].strip()))
                c["pec_format"] += ok
                c["pec_example_domain"] += ok and e["text"].strip().lower().endswith(".example")
            elif tg == "date":
                fam, ok = date_check(e["text"])
                c[f"date_{lang}_{fam}"] += 1
                c["date_calendar_valid"] += ok
            elif tg == "amount":
                nota, groups = number_notation(e["text"])
                c[f"amount_{lang}_{nota}"] += 1
                c["amount_2plus_groups"] += groups >= 2
                c["amount_has_currency"] += bool(re.search(r"€|EUR|\$|USD|GBP|£|CHF", e["text"]))
            elif tg == "percent":
                nota, _ = number_notation(e["text"])
                c[f"percent_{lang}_{nota}"] += 1
        pg.update(n_missing=miss, n_ambiguous=amb, n_short=short, n_dup_listed=sum(v - 1 for v in seen.values() if v > 1),
                  ibans_unlisted=len(ibans_text - listed_ibans),
                  ibans_text_valid=sum(iban_check(i)["all_valid"] for i in ibans_text))
    thr = max(2, math.ceil(0.02 * n))
    similarity = pair_similarity(name, meta, raw_lines)
    core = {
        t: sorted(s for s, k in c.items() if k >= max(0.3 * tpl_it_pages[t], min(tpl_it_pages[t], 2)))
        for t, c in tpl_starts.items()
        if tpl_it_pages[t] >= 2
    }
    return {
        "name": name, "similarity": similarity, "pages": pages, "sent_pages": sent_pages, "sent_chars": sent_chars, "pat_pages": pat_pages, "pat_chars": pat_chars,
        "sent_example": {k: sent_example[k] for k, v in sent_pages.items() if v >= thr},
        "chars": chars, "companies": companies, "apos_words": apos_words, "ent_strings": ent_strings,
        "ent": {k: dict(v) for k, v in ent.items()}, "pos_dec": {k: dict(v) for k, v in pos_dec.items()},
        "tpl_core": core, "ent_keys": dict(ent_keys), "deg_keys": dict(deg_keys), "meta_keys": dict(meta_keys),
        "bbox_distinct": len(bbox_values), "bbox_top": bbox_values.most_common(1),
        "type_counts": dict(C(t for (t, _), v in ent_strings.items() for _ in range(v))),
    }


def pair_similarity(name, meta, raw_lines, pairs=200):
    """How alike two pages are. 200 random pairs with the same template and language, 200 with the same language and
    different templates (fixed seed). word_ratio: difflib ratio on the word sequences. lines: share of the lines of one
    page found identical in the other, as written and with every digit masked."""
    import difflib
    import random
    rng = random.Random(name)
    groups = collections.defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m.get("tpl"), m.get("lang"))].append(i)
    same_groups = [g for g in groups.values() if len(g) >= 2]
    keys = list(groups)
    mask = lambda ls: {re.sub(r"\d", "0", l) for l in ls}

    def measure(i, j):
        a, b = raw_lines[i], raw_lines[j]
        wa, wb = " ".join(a).split(), " ".join(b).split()
        return (difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio(), len(set(a) & set(b)) / max(1, len(set(a))),
                len(mask(a) & mask(b)) / max(1, len(mask(a))))

    same, diff = [], []
    for _ in range(pairs if same_groups else 0):
        same.append(measure(*rng.sample(rng.choice(same_groups), 2)))
    for _ in range(pairs):
        ka, kb = rng.sample(keys, 2)
        if ka[1] == kb[1] and ka[0] != kb[0]:
            diff.append(measure(rng.choice(groups[ka]), rng.choice(groups[kb])))
    med = lambda xs, k: round(statistics.median(x[k] for x in xs), 3) if xs else None
    q90 = lambda xs, k: round(sorted(x[k] for x in xs)[int(0.9 * len(xs))], 3) if xs else None
    return {"pairs_same_template": len(same), "pairs_other_template": len(diff),
            "same_template": {"word_ratio_median": med(same, 0), "word_ratio_p90": q90(same, 0), "identical_lines_median": med(same, 1), "identical_lines_digits_masked_median": med(same, 2)},
            "other_template": {"word_ratio_median": med(diff, 0), "word_ratio_p90": q90(diff, 0), "identical_lines_median": med(diff, 1), "identical_lines_digits_masked_median": med(diff, 2)}}


def cache_path(name):
    return CACHE / (name.replace("/", "_") + ".pkl")


def _run_set(name):
    r = analyse_set(name)
    with open(cache_path(name), "wb") as f:
        pickle.dump(r, f)
    return name


def step_text(force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SETS if force or not cache_path(s).exists()]
    with multiprocessing.Pool(WORKERS, maxtasksperchild=4) as pool:
        for name in pool.imap_unordered(_run_set, todo):
            print("text:", name, flush=True)


def load(name):
    # the cache is written by this script only (step `text`), never taken from elsewhere
    with open(cache_path(name), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- aggregation -> stats.json

def quant(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 2), "min": xs[0], "p01": q(.01), "p05": q(.05), "p25": q(.25),
            "p50": q(.5), "p75": q(.75), "p95": q(.95), "p99": q(.99), "max": xs[-1]}


def pct(a, b):
    return round(100 * a / b, 3) if b else None


def table(pages, a, b):
    t = collections.defaultdict(collections.Counter)
    for p in pages:
        t[str(p[a])][str(p[b])] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(t.items())}


def lang_share(p, kind="stop"):
    i, e = p[f"it_{kind}"], p[f"en_{kind}"]
    return None if i + e < 5 else i / (i + e)


def page_block(pages):
    """Everything that is computed from a list of page records, used for a set and for a renderer."""
    n = len(pages)
    C = collections.Counter
    S = lambda f: sum(bool(f(p)) for p in pages)
    out = {"pages": n, "lang": dict(C(p["lang"] for p in pages)), "level": dict(sorted(C(str(p["level"]) for p in pages).items())),
           "lang_x_level": table(pages, "lang", "level"), "fonts": dict(C(p["font"] for p in pages).most_common()),
           "templates": len({p["tpl"] for p in pages}), "rot90": S(lambda p: p["rot90"]),
           "small_print_px": dict(sorted(C(str(p["small_print_px"]) for p in pages).items(), key=lambda kv: (len(kv[0]), kv[0]))),
           "small_print_outside_11_17": S(lambda p: p["small_print_px"] is not None and not 11 <= p["small_print_px"] <= 17),
           "pages_with_overlays": S(lambda p: p["n_overlays"]), "id_not_consistent_with_meta": S(lambda p: not p["id_ok"] and not p["id_other_order"]),
           "id_fields_in_other_order_than_brief": S(lambda p: p["id_other_order"]),
           "pages_missing_meta_keys": dict(C(k for p in pages for k in p["missing_keys"])),
           "pages_with_nonstandard_degradation_keys": S(lambda p: p["deg_extra"]),
           "pages_missing_degradation_params": {k: S(lambda p: p["deg"][k] is None) for k in DEG_KEYS}}
    out["chars"] = quant([p["chars"] for p in pages])
    out["words"] = quant([p["words"] for p in pages])
    out["lines"] = quant([p["lines"] for p in pages])
    out["max_line_chars"] = quant([p["max_line"] for p in pages])
    out["chars_by_level"] = {str(l): quant([p["chars"] for p in pages if p["level"] == l]) for l in range(1, 6)}
    out["chars_by_lang"] = {l: quant([p["chars"] for p in pages if p["lang"] == l]) for l in ("it", "en")}
    out["pages_under_300_chars"] = S(lambda p: p["chars"] < 300)
    out["pages_under_800_chars"] = S(lambda p: p["chars"] < 800)
    out["token_budget"] = {"budget": MAX_TOKENS,
                           "est_tokens_at_2.5_chars": quant([round(p["chars"] / 2.5) for p in pages]),
                           "pages_over_at_2.5": S(lambda p: p["chars"] / 2.5 > MAX_TOKENS), "pages_over_at_2.0": S(lambda p: p["chars"] / 2.0 > MAX_TOKENS),
                           "pages_over_80pct_at_2.0": S(lambda p: p["chars"] / 2.0 > 0.8 * MAX_TOKENS),
                           "pages_over_at_1.5": S(lambda p: p["chars"] / 1.5 > MAX_TOKENS)}
    en = [p for p in pages if p["lang"] == "en"]
    it = [p for p in pages if p["lang"] == "it"]
    ls = lambda ps, kind, f: sum(1 for p in ps if (x := lang_share(p, kind)) is not None and f(x))
    out["language"] = {
        "en_pages": len(en), "it_pages": len(it),
        "en_pages_italian_stopwords_ge_50pct": ls(en, "stop", lambda x: x >= .5), "en_pages_italian_stopwords_ge_20pct": ls(en, "stop", lambda x: x >= .2),
        "it_pages_english_stopwords_ge_50pct": ls(it, "stop", lambda x: x <= .5), "it_pages_english_stopwords_ge_20pct": ls(it, "stop", lambda x: x <= .8),
        "en_pages_italian_labels_ge_50pct": ls(en, "label", lambda x: x >= .5), "en_pages_italian_labels_ge_20pct": ls(en, "label", lambda x: x >= .2),
        "it_pages_english_labels_ge_50pct": ls(it, "label", lambda x: x <= .5), "it_pages_english_labels_ge_20pct": ls(it, "label", lambda x: x <= .8),
        "pages_with_under_5_stopwords": S(lambda p: p["it_stop"] + p["en_stop"] < 5),
        "en_pages_italian_stopword_share": quant([round(100 * x, 1) for p in en if (x := lang_share(p)) is not None]),
        "it_pages_italian_stopword_share": quant([round(100 * x, 1) for p in it if (x := lang_share(p)) is not None]),
    }
    cls = C(k for p in pages for k in p["cls"])
    esc = C(k for p in pages for k in p["esc"])
    out["unicode"] = {"pages_not_nfc": S(lambda p: not p["nfc"]), "pages_changed_by_nfkc": S(lambda p: not p["nfkc"]),
                      "pages_escaped_breaks_only": S(lambda p: p["escaped_breaks"]),
                      "pages_with_literal_escape_or_markup": dict(esc), "pages_with_char_class": dict(cls.most_common()),
                      "pages_with_apostrophe_as_accent": S(lambda p: p["apos_accent"]), "apostrophe_as_accent_words": sum(p["apos_accent"] for p in pages),
                      "it_pages_with_apostrophe_as_accent": S(lambda p: p["apos_accent"] and p["lang"] == "it"),
                      "unaccented_words": sum(p["no_accent"] for p in pages), "accented_words": sum(p["yes_accent"] for p in pages),
                      "pages_with_pipe_joined_cells": S(lambda p: p["pipe_lines"]), "pages_with_empty_lines": S(lambda p: p["empty_lines"] > 0),
                      "pages_with_double_spaces": S(lambda p: p["double_space"]), "pages_with_edge_spaces": S(lambda p: p["edge_space_lines"]),
                      "pages_not_ending_with_newline": S(lambda p: not p["ends_newline"])}
    ne = [p["n_ent"] for p in pages]
    tot = sum(ne)
    out["entities_per_page"] = quant(ne)
    out["entities"] = tot
    out["pages_under_6_entities"] = S(lambda p: p["n_ent"] < 6)
    out["entity_density_per_1000_chars"] = round(1000 * tot / max(1, sum(p["chars"] for p in pages)), 2)
    out["pages_with_missing_entity"] = S(lambda p: p["n_missing"])
    out["pages_with_ambiguous_entity"] = S(lambda p: p["n_ambiguous"])
    out["pages_with_duplicate_listed_entity"] = S(lambda p: p["n_dup_listed"])
    out["ambiguous_share_per_page_pct"] = quant([round(100 * p["n_ambiguous"] / p["n_ent"], 1) for p in pages if p["n_ent"]])
    it_all = sum(p["ibans_in_text"] for p in pages)
    out["iban_like_strings_in_text"] = {"total": it_all, "not_listed_as_entity": sum(p["ibans_unlisted"] for p in pages),
                                        "fully_valid": sum(p["ibans_text_valid"] for p in pages)}
    out["duplicates"] = {"distinct_txt": len({p["txt_sha"] for p in pages}), "distinct_after_masking_digits": len({p["digit_hash"] for p in pages}),
                         "distinct_word_patterns": len({p["pattern_hash"] for p in pages})}
    return out


def degradation_block(pages):
    out = {}
    for k in DEG_KEYS:
        f = (lambda v: abs(v)) if k == "rotation_deg" else (lambda v: v)
        out[k] = {str(l): quant([f(p["deg"][k]) for p in pages if p["level"] == l and isinstance(p["deg"][k], (int, float))]) for l in range(1, 6)}
    med = lambda k, l: out[k][str(l)].get("p50")
    order = {}
    for k, sign in (("rotation_deg", 1), ("blur", 1), ("noise_sigma", 1), ("jpeg_quality", -1)):
        ms = [med(k, l) for l in range(1, 6)]
        order[k] = None if any(m is None for m in ms[1:]) else all(sign * (b - a) >= 0 for a, b in zip(ms[1:], ms[2:])) and sign * (ms[4] - ms[1]) > 0
    out["medians_ordered_L2_to_L5"] = order
    l1 = [p for p in pages if p["level"] == 1]
    out["L1_pages_with_any_degradation"] = sum(any(isinstance(p["deg"][k], (int, float)) and ((k == "jpeg_quality" and 0 < p["deg"][k] < 95) or (k != "jpeg_quality" and abs(p["deg"][k]) > 0)) for k in DEG_KEYS) for p in l1)
    # overlap: share of L4 pages whose noise is above the median of L5, i.e. levels that cannot be told apart
    for a, b in ((2, 3), (3, 4), (4, 5)):
        xa = sorted(p["deg"]["noise_sigma"] for p in pages if p["level"] == a and isinstance(p["deg"]["noise_sigma"], (int, float)))
        xb = sorted(p["deg"]["noise_sigma"] for p in pages if p["level"] == b and isinstance(p["deg"]["noise_sigma"], (int, float)))
        out[f"noise_L{a}_pages_at_or_above_L{b}_minimum_pct"] = pct(sum(x >= xb[0] for x in xa), len(xa)) if xa and xb else None
    return out


def entity_block(ents, pos):
    """ents: type group -> Counter. Rates over the entities present in the ground truth unless said otherwise."""
    out = {}
    for t, c in sorted(ents.items(), key=lambda kv: -kv[1]["n"]):
        n = c["n"]
        ok = n - c.get("missing", 0)
        r = {"n": n, "missing_from_ground_truth": c.get("missing", 0)}
        for k in ("short", "multi", "sub", "ambiguous", "dup_listed", "in_next_page", "first3", "last3", "nonascii"):
            r[k] = c.get(k, 0)
            r[k + "_pct"] = pct(c.get(k, 0), ok)
        r["body_pct"] = pct(ok - c.get("first3", 0) - c.get("last3", 0), ok)
        r["position_deciles"] = [pos.get(t, {}).get(i, 0) for i in range(10)]
        r["chars"] = {k[4:]: v for k, v in c.items() if k.startswith("has_")}
        r["detail"] = {k: v for k, v in sorted(c.items()) if re.match(r"(tail_)?(iban|vat|cf|cig|cup|pec|date|amount|percent)_", k)}
        r["small_print_flag"] = c.get("small_print_flag", 0)
        out[t] = r
    return out


def validity_block(eb):
    g = lambda t, k: eb.get(t, {}).get("detail", {}).get(k, 0)
    n = lambda t: eb.get(t, {}).get("n", 0)
    v = {}
    if n("iban"):
        N = n("iban")
        v["iban"] = {"n": N, "shape_IT_plus_25_pct": pct(g("iban", "iban_shape"), N), "italian_layout_with_cin_letter_pct": pct(g("iban", "iban_format"), N),
                     "mod97_valid_pct": pct(g("iban", "iban_mod97"), N), "cin_valid_pct": pct(g("iban", "iban_cin"), N),
                     "fully_valid_pct": pct(g("iban", "iban_all_valid"), N), "written_in_groups_of_4_pct": pct(g("iban", "iban_grouped4"), N)}
    if n("vat"):
        v["vat"] = {"n": n("vat"), "format_11_digits_pct": pct(g("vat", "vat_format"), n("vat")), "check_digit_valid_pct": pct(g("vat", "vat_check"), n("vat"))}
    if n("cf"):
        N = n("cf")
        v["cf"] = {"n": N, "shape_16_pct": pct(g("cf", "cf_shape"), N), "month_letter_valid_pct": pct(g("cf", "cf_month_ok"), N),
                   "day_field_valid_pct": pct(g("cf", "cf_day_ok"), N), "check_char_valid_pct": pct(g("cf", "cf_check"), N),
                   "fully_valid_pct": pct(g("cf", "cf_all_valid"), N), "numeric_11_digits": g("cf", "cf_numeric")}
    if n("cig"):
        v["cig"] = {"n": n("cig"), "format_10_alnum_pct": pct(g("cig", "cig_format"), n("cig")), "shape_7_plus_3_hex_pct": pct(g("cig", "cig_real_shape"), n("cig"))}
    if n("cup"):
        v["cup"] = {"n": n("cup"), "format_pct": pct(g("cup", "cup_format"), n("cup"))}
    if n("pec"):
        v["pec"] = {"n": n("pec"), "address_format_pct": pct(g("pec", "pec_format"), n("pec")), "domain_example_pct": pct(g("pec", "pec_example_domain"), n("pec"))}
    if n("date"):
        d = eb["date"]["detail"]
        fam = lambda lang: {k.split("_", 2)[2]: x for k, x in d.items() if k.startswith(f"date_{lang}_")}
        v["date"] = {"n": n("date"), "calendar_valid_pct": pct(d.get("date_calendar_valid", 0), n("date")), "it_pages": fam("it"), "en_pages": fam("en")}
    for t in ("amount", "percent"):
        if n(t):
            d = eb[t]["detail"]
            fam = lambda lang: {k.split("_", 2)[2]: x for k, x in d.items() if k.startswith(f"{t}_{lang}_")}
            v[t] = {"n": n(t), "it_pages": fam("it"), "en_pages": fam("en")}
            i, e = v[t]["it_pages"], v[t]["en_pages"]
            v[t]["it_pages_in_english_notation_pct"] = pct(i.get("en", 0), sum(i.values()))
            v[t]["en_pages_in_italian_notation_pct"] = pct(e.get("it", 0), sum(e.values()))
    if n("amount"):
        v["amount"]["two_or_more_thousands_groups_pct"] = pct(g("amount", "amount_2plus_groups"), n("amount"))
        v["amount"]["with_currency_pct"] = pct(g("amount", "amount_has_currency"), n("amount"))
    o = eb.get("other", {}).get("detail", {})
    if o:
        v["tail_types_named_like_codes"] = {
            "iban": {"n": o.get("tail_iban_n", 0), "mod97_valid_pct": pct(o.get("tail_iban_mod97", 0), o.get("tail_iban_n", 0))},
            "vat": {"n": o.get("tail_vat_n", 0), "check_digit_valid_pct": pct(o.get("tail_vat_check", 0), o.get("tail_vat_n", 0))},
            "cf": {"n": o.get("tail_cf_n", 0), "check_char_valid_pct": pct(o.get("tail_cf_check", 0), o.get("tail_cf_n", 0))}}
    return v


def merge(dst, src):
    for k, c in src.items():
        dst[k].update(c)


def sentence_block(sent_pages, sent_chars, pat_pages, pat_chars, n_pages):
    tc, pc = sum(sent_chars.values()), sum(pat_chars.values())
    thr1 = max(2, math.ceil(0.01 * n_pages))
    thr2 = max(2, math.ceil(0.02 * n_pages))
    return {
        "sentence_occurrences_30plus_chars": sum(sent_pages.values()), "distinct_sentences": len(sent_pages),
        "distinct_word_patterns": len(pat_pages), "pattern_occurrences": sum(pat_pages.values()),
        "chars_in_sentences_seen_on_2plus_pages_pct": pct(sum(sent_chars[k] for k, v in sent_pages.items() if v >= 2), tc),
        "chars_in_sentences_seen_on_1pct_of_pages_pct": pct(sum(sent_chars[k] for k, v in sent_pages.items() if v >= thr1), tc),
        "sentences_on_more_than_2pct_of_pages": sum(v > 0.02 * n_pages for v in sent_pages.values()),
        "max_pages_one_sentence": max(sent_pages.values(), default=0),
        "max_pages_one_sentence_pct": pct(max(sent_pages.values(), default=0), n_pages),
        "chars_in_patterns_seen_on_2plus_pages_pct": pct(sum(pat_chars[k] for k, v in pat_pages.items() if v >= 2), pc),
        "chars_in_patterns_seen_on_1pct_of_pages_pct": pct(sum(pat_chars[k] for k, v in pat_pages.items() if v >= thr1), pc),
        "chars_in_patterns_seen_on_2pct_of_pages_pct": pct(sum(pat_chars[k] for k, v in pat_pages.items() if v >= thr2), pc),
        "patterns_covering_half_of_pattern_chars": _half(pat_chars),
    }


def _half(chars):
    tot, acc = sum(chars.values()), 0
    for i, (_, v) in enumerate(chars.most_common(), 1):
        acc += v
        if acc >= tot / 2:
            return i
    return 0


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def step_stats():
    C = collections.Counter
    stats = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "root": str(ROOT),
             "scope": "meta.json and .txt of 63 sets. No PNG was opened.", "sets": {}, "renderers": {}, "cross": {}}
    R = {r: {"pages": [], "ent": collections.defaultdict(C), "pos": collections.defaultdict(C), "sent_pages": C(), "sent_chars": C(),
             "pat_pages": C(), "pat_chars": C(), "chars": C(), "companies": C(), "apos": C(), "ent_strings": C(), "types": C(),
             "tpl_sets": collections.defaultdict(list), "tpl_core": {}, "seeds": collections.defaultdict(list), "examples": {},
             "ent_keys": C(), "deg_keys": C(), "txt": collections.defaultdict(list), "digit": collections.defaultdict(list), "pat_by_set": {},
             "sent_by_set": {}, "ent_by_set": {}, "type_sets": collections.defaultdict(set)} for r in RENDERERS}
    for name in SETS:
        d = load(name)
        r = name.split("/")[0]
        A = R[r]
        pages = d["pages"]
        for p in pages:
            p["set"] = name
        s = page_block(pages)
        s["role"] = role_of(name)
        s["degradation"] = degradation_block(pages)
        eb = entity_block({k: C(v) for k, v in d["ent"].items()}, d["pos_dec"])
        s["entity_types"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("detail", "position_deciles", "chars")} for k, v in eb.items()}
        s["validity"] = validity_block(eb)
        s["distinct_entity_types"] = len(d["type_counts"])
        s["tpl_x_lang"] = table(pages, "tpl", "lang")
        s["tpl_x_level"] = table(pages, "tpl", "level")
        cells = [s["tpl_x_lang"][t].get(l, 0) for t in s["tpl_x_lang"] for l in ("it", "en")]
        s["tpl_x_lang_empty_cells"] = sum(c == 0 for c in cells)
        cells = [s["tpl_x_level"][t].get(str(l), 0) for t in s["tpl_x_level"] for l in range(1, 6)]
        s["tpl_x_level_empty_cells"] = sum(c == 0 for c in cells)
        s["tpl_x_level_min_max_cell"] = [min(cells), max(cells)]
        tp = C(p["tpl"] for p in pages)
        s["pages_per_template_min_max"] = [min(tp.values()), max(tp.values())]
        seeds = [p["seed"] for p in pages]
        s["seeds"] = {"unique": len(set(seeds)), "base": min(seeds), "follow_seed_times_1000_plus_index": all(p["seed"] - p["idx"] == seeds[0] - pages[0]["idx"] and p["seed"] % 1000 == p["idx"] % 1000 for p in pages) and seeds[0] % 1000 == 0,
                      "consecutive": sorted(seeds) == list(range(min(seeds), min(seeds) + len(seeds)))}
        s["sentences"] = sentence_block(d["sent_pages"], d["sent_chars"], d["pat_pages"], d["pat_chars"], len(pages))
        s["similarity"] = d["similarity"]
        s["meta_keys"] = d["meta_keys"]
        s["entity_keys"] = d["ent_keys"]
        s["degradation_keys"] = d["deg_keys"]
        s["bbox_distinct_values"] = d["bbox_distinct"]
        s["bbox_most_common"] = [list(d["bbox_top"][0][0]), d["bbox_top"][0][1]] if d["bbox_top"] else None
        s["wrong_page_recall_pct"] = pct(sum(v.get("in_next_page", 0) for v in d["ent"].values()), sum(v["n"] - v.get("missing", 0) for v in d["ent"].values()))
        stats["sets"][name] = s
        A["pages"] += pages
        merge(A["ent"], d["ent"])
        merge(A["pos"], d["pos_dec"])
        for k in ("sent_pages", "sent_chars", "pat_pages", "pat_chars", "chars", "companies", "ent_strings"):
            A[k].update(d[k])
        A["apos"].update(d["apos_words"])
        A["types"].update(d["type_counts"])
        for t in d["type_counts"]:
            A["type_sets"][t].add(name[-1])
        A["ent_keys"].update(d["ent_keys"])
        A["deg_keys"].update(d["deg_keys"])
        A["pat_by_set"][name] = d["pat_pages"]
        A["ent_by_set"][name] = d["ent_strings"]
        A["sent_by_set"][name] = set(d["sent_pages"])
        if role_of(name) != "test":
            A["examples"].update(d["sent_example"])
        for t in {p["tpl"] for p in pages}:
            A["tpl_sets"][t].append(name[-1])
        for t, core in d["tpl_core"].items():
            A["tpl_core"][(name[-1], t)] = core
        for p in pages:
            A["seeds"][p["seed"]].append(name)
            A["txt"][p["txt_sha"]].append(name)
            A["digit"][p["digit_hash"]].append(name)
        del d

    for r, A in R.items():
        pages = A["pages"]
        s = page_block(pages)
        s["sets"] = len(RENDERERS[r])
        s["degradation"] = degradation_block(pages)
        s["degradation_sets_not_ordered"] = {k: [n for n in SETS if n.startswith(r + "/") and stats["sets"][n]["degradation"]["medians_ordered_L2_to_L5"][k] is False] for k in DEG_KEYS}
        eb = entity_block(A["ent"], A["pos"])
        s["entity_types"] = eb
        s["validity"] = validity_block(eb)
        tot_ok = sum(v["n"] - v["missing_from_ground_truth"] for v in eb.values())
        tot = sum(v["n"] for v in eb.values())
        s["entity_totals"] = {"n": tot, **{k: sum(v[k] for v in eb.values()) for k in ("missing_from_ground_truth", "short", "multi", "sub", "ambiguous", "dup_listed", "in_next_page", "first3", "last3", "nonascii")}}
        s["entity_totals"].update({k + "_pct": pct(s["entity_totals"][k], tot_ok) for k in ("short", "multi", "sub", "ambiguous", "dup_listed", "in_next_page", "first3", "last3", "nonascii")})
        s["distinct_entity_types"] = len(A["types"])
        s["types_outside_the_brief"] = {"types": len([t for t in A["types"] if t not in CANON]), "entities": sum(v for t, v in A["types"].items() if t not in CANON),
                                        "entities_pct": pct(sum(v for t, v in A["types"].items() if t not in CANON), tot),
                                        "count_of_counts": dict(C(v for t, v in A["types"].items() if t not in CANON).most_common(8)),
                                        "types_in_one_set_only": sum(len(A["type_sets"][t]) == 1 for t in A["types"] if t not in CANON),
                                        "top": [[t, v] for t, v in A["types"].most_common() if t not in CANON][:25]}
        # repetition of entity strings
        by_type = collections.defaultdict(C)
        for (t, x), v in A["ent_strings"].items():
            by_type[t if t in CANON else "other"][x] += v
        s["entity_repetition"] = {t: {"occurrences": sum(c.values()), "distinct": len(c), "uniqueness_pct": pct(len(c), sum(c.values())),
                                      "max_repeats_one_string": max(c.values()), "strings_used_10plus_times": sum(v >= 10 for v in c.values()),
                                      "occurrences_of_strings_used_10plus_times_pct": pct(sum(v for v in c.values() if v >= 10), sum(c.values()))}
                                  for t, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))}
        A["by_type"] = by_type
        s["sentences"] = sentence_block(A["sent_pages"], A["sent_chars"], A["pat_pages"], A["pat_chars"], len(pages))
        s["top_repeated_sentences_train_dev_only"] = [[A["sent_pages"][k], A["examples"][k]] for k in sorted(A["examples"], key=lambda k: -A["sent_pages"][k])[:12]]
        # templates
        names = A["tpl_sets"]
        shared = {t: "".join(sorted(v)) for t, v in names.items() if len(v) > 1}
        s["template_names"] = {"distinct": len(names), "sum_over_sets": sum(len(v) for v in names.values()), "shared_by_more_than_one_set": len(shared),
                               "shared": dict(sorted(shared.items())) if len(shared) <= 80 else dict(sorted(shared.items())[:80]),
                               "per_set": {n[-1]: stats["sets"][n]["templates"] for n in SETS if n.startswith(r + "/")}}
        same, diff = [], []
        keys = sorted(A["tpl_core"])
        for i, (sa, ta) in enumerate(keys):
            for sb, tb in keys[i + 1:]:
                if sa == sb:
                    continue
                if ta == tb:
                    same.append((jaccard(A["tpl_core"][(sa, ta)], A["tpl_core"][(sb, tb)]), ta, sa, sb))
        import random
        rng = random.Random(1)
        for _ in range(4000 if len(keys) > 1 else 0):
            (sa, ta), (sb, tb) = rng.sample(keys, 2)
            if ta != tb:
                diff.append(jaccard(A["tpl_core"][(sa, ta)], A["tpl_core"][(sb, tb)]))
        wk = [(jaccard(A["tpl_core"][a], A["tpl_core"][b])) for i, a in enumerate(keys) for b in keys[i + 1:] if a[0] == b[0] and a[1] != b[1]]
        per_tpl = collections.defaultdict(list)
        for j, t, sa, sb in same:
            if j is not None:
                per_tpl[t].append(j)
        s["template_skeletons"] = {
            "method": "Italian pages; core skeleton of a (set, template) = line starts (first 3 words, digits masked) present on >= 30% of its pages; Jaccard between cores",
            "same_name_different_set_jaccard": quant([round(j, 3) for j, *_ in same if j is not None]),
            "different_name_different_set_jaccard": quant([round(j, 3) for j in diff if j is not None]),
            "different_name_same_set_jaccard": quant([round(j, 3) for j in wk if j is not None]),
            "same_name_pairs_identical_core": sum(j == 1 for j, *_ in same), "same_name_pairs_disjoint_core": sum(j == 0 for j, *_ in same),
            "per_shared_name_mean_jaccard": {t: round(statistics.fmean(v), 3) for t, v in sorted(per_tpl.items())} if len(per_tpl) <= 80 else None,
            "core_size": quant([len(v) for v in A["tpl_core"].values()])}
        s["seed_values_used_by_more_than_one_set"] = sum(len(v) > 1 for v in A["seeds"].values())
        dup_txt = {k: v for k, v in A["txt"].items() if len(v) > 1}
        dup_dig = {k: v for k, v in A["digit"].items() if len(v) > 1}
        s["duplicates"].update({
            "identical_txt_groups": len(dup_txt), "pages_in_identical_txt_groups": sum(len(v) for v in dup_txt.values()),
            "identical_txt_groups_within_one_set": sum(len(set(v)) == 1 for v in dup_txt.values()),
            "identical_txt_groups_across_sets": sum(len(set(v)) > 1 for v in dup_txt.values()),
            "digit_masked_groups": len(dup_dig), "pages_in_digit_masked_groups": sum(len(v) for v in dup_dig.values()),
            "largest_digit_masked_group": max(map(len, dup_dig.values()), default=1)})
        co = A["companies"]
        s["companies"] = {"mentions": sum(co.values()), "distinct": len(co), "top": co.most_common(10),
                          "names_covering_half_of_mentions": _half(co)}
        s["non_ascii_chars"] = [[f"U+{ord(ch):04X}", unicodedata.name(ch, "?"), n] for ch, n in A["chars"].most_common(60)]
        s["apostrophe_as_accent_top_words"] = A["apos"].most_common(25)
        s["meta_keys"] = {k: v for k, v in C(k for n in SETS if n.startswith(r + "/") for k in stats["sets"][n]["meta_keys"] for _ in range(stats["sets"][n]["meta_keys"][k])).items()}
        s["entity_keys"] = dict(A["ent_keys"])
        s["degradation_keys"] = dict(A["deg_keys"])
        s["chars_hist_250"] = dict(sorted(C(p["chars"] // 250 * 250 for p in pages).items()))
        s["entities_hist_5"] = dict(sorted(C(p["n_ent"] // 5 * 5 for p in pages).items()))
        stats["renderers"][r] = s

    # ---- across renderers
    X = stats["cross"]
    X["seed_values_shared_between_renderers"] = {f"{a}~{b}": len(set(R[a]["seeds"]) & set(R[b]["seeds"])) for a, b in (("gemini", "spark"), ("gemini", "chatgpt"), ("spark", "chatgpt"))}
    X["identical_txt_between_renderers"] = {f"{a}~{b}": len(set(R[a]["txt"]) & set(R[b]["txt"])) for a, b in (("gemini", "spark"), ("gemini", "chatgpt"), ("spark", "chatgpt"))}
    X["template_names_shared_between_renderers"] = {f"{a}~{b}": sorted(set(R[a]["tpl_sets"]) & set(R[b]["tpl_sets"])) for a, b in (("gemini", "spark"), ("gemini", "chatgpt"), ("spark", "chatgpt"))}
    train_strings = collections.defaultdict(set)
    for (t, x) in R["gemini"]["ent_strings"]:
        train_strings[t if t in CANON else "other"].add(x)
    train_any = set().union(*train_strings.values())
    leak = {}
    for r in ("spark", "chatgpt"):
        leak[r] = {}
        for t, c in R[r]["by_type"].items():
            sh = [x for x in c if x in train_any]
            leak[r][t] = {"distinct": len(c), "distinct_also_in_train": len(sh), "distinct_also_in_train_pct": pct(len(sh), len(c)),
                          "occurrences": sum(c.values()), "occurrences_also_in_train_pct": pct(sum(c[x] for x in sh), sum(c.values()))}
        allc = C()
        for c in R[r]["by_type"].values():
            allc.update(c)
        sh = [x for x in allc if x in train_any]
        leak[r]["ALL"] = {"distinct": len(allc), "distinct_also_in_train": len(sh), "distinct_also_in_train_pct": pct(len(sh), len(allc)),
                          "occurrences": sum(allc.values()), "occurrences_also_in_train_pct": pct(sum(allc[x] for x in sh), sum(allc.values()))}
    X["entity_strings_eval_also_in_train"] = leak
    sl = {}
    for r in ("spark", "chatgpt"):
        A, G = R[r], R["gemini"]
        sl[r] = {"sentence_occurrences": sum(A["sent_pages"].values()),
                 "occurrences_identical_in_train_pct": pct(sum(v for k, v in A["sent_pages"].items() if k in G["sent_pages"]), sum(A["sent_pages"].values())),
                 "distinct_identical_in_train": sum(k in G["sent_pages"] for k in A["sent_pages"]),
                 "pattern_occurrences": sum(A["pat_pages"].values()),
                 "pattern_occurrences_also_in_train_pct": pct(sum(v for k, v in A["pat_pages"].items() if k in G["pat_pages"]), sum(A["pat_pages"].values())),
                 "distinct_patterns": len(A["pat_pages"]), "distinct_patterns_also_in_train": sum(k in G["pat_pages"] for k in A["pat_pages"])}
        co = {metrics.squash(k): k for k in A["companies"]}
        cg = {metrics.squash(k) for k in G["companies"]}
        both = [co[k] for k in co if k in cg]
        sl[r]["company_names_distinct"] = len(co)
        sl[r]["company_names_also_in_train"] = len(both)
        sl[r]["company_mentions_also_in_train_pct"] = pct(sum(A["companies"][k] for k in both), sum(A["companies"].values()))
        sl[r]["company_names_also_in_train_examples"] = sorted(both, key=lambda k: -A["companies"][k])[:10]
    sl["spark~chatgpt"] = {"pattern_occurrences_of_chatgpt_also_in_spark_pct": pct(sum(v for k, v in R["chatgpt"]["pat_pages"].items() if k in R["spark"]["pat_pages"]), sum(R["chatgpt"]["pat_pages"].values()))}
    X["sentences_eval_also_in_train"] = sl
    # dev vs test inside the evaluation renderers, and set-to-set overlap of sentence patterns
    ov = {}
    for r, A in R.items():
        names = list(A["pat_by_set"])
        m = {}
        for a in names:
            ta = sum(A["pat_by_set"][a].values())
            m[a[-1]] = {b[-1]: (round(100 * sum(v for k, v in A["pat_by_set"][a].items() if k in A["pat_by_set"][b]) / ta, 1) if ta else None) for b in names if b != a}
        vals = [v for row in m.values() for v in row.values() if v is not None]
        ov[r] = {"mean_pct_of_pattern_occurrences_of_a_set_found_in_another_set": round(statistics.fmean(vals), 1), "min": min(vals), "max": max(vals), "matrix": m}
        dev = [n for n in names if role_of(n) == "dev"]
        test = [n for n in names if role_of(n) == "test"]
        if dev and test:
            devp = set().union(*(A["pat_by_set"][n].keys() for n in dev))
            devs = set().union(*(A["sent_by_set"][n] for n in dev))
            ov[r]["test_sets_vs_dev_sets"] = {
                n: {"pattern_occurrences_also_in_dev_pct": pct(sum(v for k, v in A["pat_by_set"][n].items() if k in devp), sum(A["pat_by_set"][n].values())),
                    "distinct_sentences_identical_in_dev_pct": pct(len(A["sent_by_set"][n] & devs), len(A["sent_by_set"][n]))} for n in test}
    X["sentence_pattern_overlap_between_sets"] = ov
    el = {}
    for r, A in R.items():
        dev = [n for n in A["ent_by_set"] if role_of(n) == "dev"]
        test = [n for n in A["ent_by_set"] if role_of(n) == "test"]
        if not (dev and test):
            continue
        devs = {x for n in dev for (_, x) in A["ent_by_set"][n]}
        for n in test:
            bt = collections.defaultdict(lambda: [0, 0])
            for (t, x), v in A["ent_by_set"][n].items():
                bt[t][0] += v
                bt[t][1] += v * (x in devs)
            el[n] = {t: {"occurrences": a, "also_in_dev_sets_pct": pct(b, a)} for t, (a, b) in sorted(bt.items())}
            el[n]["ALL"] = {"occurrences": sum(a for a, _ in bt.values()), "also_in_dev_sets_pct": pct(sum(b for _, b in bt.values()), sum(a for a, _ in bt.values()))}
    X["entity_strings_test_also_in_dev_same_renderer"] = el
    X["type_mix_pct"] = {r: {t: pct(R[r]["ent"][t]["n"], sum(v["n"] for v in R[r]["ent"].values())) for t in CANON + ["other"]} for r in R}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("stats:", OUT / "stats.json", (OUT / "stats.json").stat().st_size, "bytes")


# ---------------------------------------------------------------- report (Italian), inline SVG charts

COL = {"gemini": "#2a78d6", "spark": "#eb6834", "chatgpt": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"


def fi(x):
    """Italian integer: 39.000"""
    return f"{x:,.0f}".replace(",", ".") if isinstance(x, (int, float)) else "–"


def fp(x, d=1):
    """Italian percentage: 22,8%"""
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",") + "%"


def fd(x, d=2):
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo, hi, n=5):
    span = (hi - lo) or 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    out = []
    while t0 <= hi + 1e-9:
        out.append(round(t0, 6))
        t0 += step
    return out


def tick_label(v):
    return fi(v) if abs(v) >= 1000 or float(v).is_integer() else f"{v:g}".replace(".", ",")


def legend(items, x, y):
    out, cx = [], x
    for name, color in items:
        out.append(f'<rect x="{cx}" y="{y - 9}" width="12" height="12" rx="2" fill="{color}"/><text x="{cx + 17}" y="{y + 1}" font-size="11" fill="{INK2}">{esc(name)}</text>')
        cx += 30 + 6.3 * len(name)
    return "".join(out)


def frame(w, h, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" font-family="system-ui,-apple-system,Helvetica,Arial,sans-serif">{body}</svg>'


def line_chart(series, xlabel, ylabel, w=720, h=280, ymax=None, xticks=None, band=None, ymin=0):
    """series: [(name, color, [(x, y), ...])]. One y axis. band: {name: [(x, lo, hi)]} drawn as a min-max range."""
    L, R, T, B = 58, 16, 30, 46
    xs = [x for _, _, pts in series for x, _ in pts]
    ys = [y for _, _, pts in series for _, y in pts] + ([v for b in band.values() for _, lo, hi in b for v in (lo, hi)] if band else [])
    x0, x1 = min(xs), max(xs)
    y1 = ymax or max(ys) * 1.08
    X = lambda v: L + (v - x0) / ((x1 - x0) or 1) * (w - L - R)
    Y = lambda v: h - B - (v - ymin) / ((y1 - ymin) or 1) * (h - T - B)
    o = []
    for t in nice_ticks(ymin, y1):
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" font-size="10.5" text-anchor="end" fill="{MUTED}">{tick_label(t)}</text>')
    for t in (xticks or nice_ticks(x0, x1, 8)):
        o.append(f'<text x="{X(t):.1f}" y="{h - B + 15}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{tick_label(t)}</text>')
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" stroke="{MUTED}"/>')
    if band:
        for name, color, _ in series:
            for x, lo, hi in band.get(name, []):
                o.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" stroke="{color}" stroke-width="5" stroke-opacity=".22" stroke-linecap="round"/>')
    for name, color, pts in series:
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        if len(pts) <= 12:
            o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="2"/>' for x, y in pts]
    o.append(legend([(n, c) for n, c, _ in series], L, 14))
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 8}" font-size="11" text-anchor="middle" fill="{INK2}">{esc(xlabel)}</text>')
    o.append(f'<text transform="translate(13,{(T + h - B) / 2}) rotate(-90)" font-size="11" text-anchor="middle" fill="{INK2}">{esc(ylabel)}</text>')
    return frame(w, h, "".join(o))


def set_bars(values, ylabel, xlabel, w=720, h=250, ymax=None, labels=True, fmt=fd):
    """values: {set name: value}. One bar per set, coloured by renderer, in the order gemini, spark, chatgpt."""
    L, R, T, B = 58, 12, 30, 48
    names = [n for n in SETS if values.get(n) is not None]
    y1 = ymax or max(values[n] for n in names) * 1.1
    bw = (w - L - R - 24) / len(names)
    Y = lambda v: h - B - v / y1 * (h - T - B)
    o = []
    for t in nice_ticks(0, y1):
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" font-size="10.5" text-anchor="end" fill="{MUTED}">{tick_label(t)}</text>')
    x, prev = L, None
    for n in names:
        r = n.split("/")[0]
        if prev and r != prev:
            x += 12
        prev = r
        v = values[n]
        top = Y(v)
        hh = h - B - top
        o.append(f'<path d="M{x + 1:.1f},{h - B} v{-max(hh - 2, 0):.1f} q0,-2 2,-2 h{bw - 6:.1f} q2,0 2,2 v{max(hh - 2, 0):.1f} z" fill="{COL[r]}"/>')
        o.append(f'<text x="{x + bw / 2:.1f}" y="{h - B + 12}" font-size="9" text-anchor="middle" fill="{MUTED}">{n[-1]}</text>')
        x += bw
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" stroke="{MUTED}"/>')
    o.append(legend([(r, COL[r]) for r in RENDERERS], L, 14))
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 8}" font-size="11" text-anchor="middle" fill="{INK2}">{esc(xlabel)}</text>')
    o.append(f'<text transform="translate(13,{(T + h - B) / 2}) rotate(-90)" font-size="11" text-anchor="middle" fill="{INK2}">{esc(ylabel)}</text>')
    return frame(w, h, "".join(o))


def grouped_hbars(cats, series, xlabel, w=720, xmax=None, fmt=fp):
    """cats: labels; series: [(name, color, [values])]. Horizontal bars, one group per category, value at the bar end."""
    L, R, T, B = 78, 50, 30, 40
    bh, gap = 9, 9
    gh = len(series) * (bh + 2) + gap
    h = T + B + gh * len(cats)
    x1 = xmax or max(v or 0 for _, _, vs in series for v in vs) * 1.08
    X = lambda v: L + v / x1 * (w - L - R)
    o = []
    for t in nice_ticks(0, x1, 6):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{h - B}" stroke="{GRID}"/><text x="{X(t):.1f}" y="{h - B + 14}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{tick_label(t)}</text>')
    for i, c in enumerate(cats):
        y = T + i * gh + gap / 2
        o.append(f'<text x="{L - 8}" y="{y + gh / 2 - 2:.1f}" font-size="11" text-anchor="end" fill="{INK}">{esc(c)}</text>')
        for k, (_, color, vs) in enumerate(series):
            v = vs[i] or 0
            yy = y + k * (bh + 2)
            ww = max(X(v) - L, 0)
            if ww > 0.2:
                o.append(f'<path d="M{L},{yy} h{max(ww - 2, 0):.1f} q2,0 2,2 v{bh - 4} q0,2 -2,2 h{-max(ww - 2, 0):.1f} z" fill="{color}"/>')
            o.append(f'<text x="{X(v) + 4:.1f}" y="{yy + bh - 1}" font-size="9" fill="{INK2}">{fmt(vs[i])}</text>')
    o.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{h - B}" stroke="{MUTED}"/>')
    o.append(legend([(n, c) for n, c, _ in series], L, 14))
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 8}" font-size="11" text-anchor="middle" fill="{INK2}">{esc(xlabel)}</text>')
    return frame(w, h, "".join(o))


def stacked_rows(rows, segs, xlabel, w=720):
    """rows: [(label, [v...])] summing to 100; segs: [(name, color)]."""
    L, R, T, B = 150, 16, 30, 40
    bh = 15
    h = T + B + (bh + 7) * len(rows)
    X = lambda v: L + v / 100 * (w - L - R)
    o = []
    for t in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{h - B}" stroke="{GRID}"/><text x="{X(t):.1f}" y="{h - B + 14}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{t}</text>')
    for i, (label, vs) in enumerate(rows):
        y = T + i * (bh + 7) + 3
        o.append(f'<text x="{L - 8}" y="{y + bh - 4}" font-size="11" text-anchor="end" fill="{INK}">{esc(label)}</text>')
        acc = 0
        for (name, color), v in zip(segs, vs):
            ww = X(acc + v) - X(acc)
            if ww > 0.3:
                o.append(f'<rect x="{X(acc) + 1:.1f}" y="{y}" width="{max(ww - 2, 0.3):.1f}" height="{bh}" fill="{color}"/>')
            if ww > 34:
                tc = "#fff" if color not in ("#dcdbd5", "#eda100") else INK
                o.append(f'<text x="{X(acc) + ww / 2:.1f}" y="{y + bh - 4}" font-size="9.5" text-anchor="middle" fill="{tc}">{fp(v)}</text>')
            acc += v
    o.append(legend(segs, L, 14))
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 8}" font-size="11" text-anchor="middle" fill="{INK2}">{esc(xlabel)}</text>')
    return frame(w, h, "".join(o))


def tbl(head, rows, num_from=1, cls=""):
    th = "".join(f'<th class="{"l" if i < num_from else "r"}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"l" if i < num_from else "r"}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def fig(svg, caption):
    return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'


CSS = """
@page { size: A4; margin: 13mm 12mm 14mm 12mm; }
* { box-sizing: border-box; }
body { font: 10.4px/1.45 system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif; color: #0b0b0b; background: #fff; margin: 0 auto; max-width: 186mm; }
h1 { font-size: 21px; margin: 0 0 2px; } h2 { font-size: 15px; margin: 22px 0 6px; border-bottom: 1.5px solid #0b0b0b; padding-bottom: 3px; break-after: avoid; }
h3 { font-size: 11.8px; margin: 14px 0 4px; break-after: avoid; } p { margin: 4px 0 6px; } .sub { color: #52514e; margin-bottom: 10px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9.2px; font-variant-numeric: tabular-nums; break-inside: auto; }
th, td { padding: 2.2px 5px; border-bottom: 1px solid #e4e3df; vertical-align: top; } th { border-bottom: 1px solid #898781; font-weight: 600; color: #52514e; }
tr { break-inside: avoid; } .l { text-align: left; } .r { text-align: right; } table.tight td, table.tight th { padding: 1.5px 3px; font-size: 8.3px; }
figure { margin: 8px 0 12px; break-inside: avoid; } figcaption { font-size: 9px; color: #52514e; margin-top: 2px; }
.verdict { border: 1.5px solid #0b0b0b; padding: 8px 11px; margin: 8px 0 10px; } .verdict p { margin: 3px 0; }
.sev { display: inline-block; min-width: 50px; text-align: center; font-size: 8.4px; font-weight: 700; padding: 0 4px; border-radius: 3px; border: 1px solid; }
.s1 { color: #a11; border-color: #d03b3b; background: #fbe9e9; } .s2 { color: #8a3d12; border-color: #ec835a; background: #fdf0ea; }
.s3 { color: #6b4b00; border-color: #fab219; background: #fff6df; } .s4 { color: #52514e; border-color: #898781; background: #f3f2ee; }
ol.def { padding-left: 0; list-style: none; margin: 4px 0; } ol.def li { margin: 0 0 5px; padding-left: 62px; text-indent: -62px; break-inside: avoid; }
ol.def li .sev { text-indent: 0; margin-right: 8px; } code { font: 9.3px ui-monospace,Menlo,monospace; background: #f3f2ee; padding: 0 2px; border-radius: 2px; }
.pb { break-before: page; } .ok { color: #0a6b0a; font-weight: 600; } .bad { color: #a11; font-weight: 600; } .note { color: #52514e; font-size: 9.2px; }
ul { margin: 4px 0 6px; padding-left: 16px; } li { margin-bottom: 2px; }
"""


def build(pdf=None):
    S = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    D = json.loads((OUT / "digests.json").read_text(encoding="utf-8"))
    RR, ST, X = S["renderers"], S["sets"], S["cross"]
    G, K, H = RR["gemini"], RR["spark"], RR["chatgpt"]
    rn = list(RENDERERS)
    sets_of = lambda r: [n for n in SETS if n.startswith(r + "/")]
    sev = lambda k: f'<span class="sev s{k}">{["", "CRITICO", "ALTO", "MEDIO", "BASSO"][k]}</span>'
    et = lambda r, t: RR[r]["entity_types"].get(t, {})
    amb_set = lambda n: pct(sum(e["ambiguous"] for e in ST[n]["entity_types"].values()), ST[n]["entities"])
    other_set = lambda n: pct(ST[n]["entity_types"].get("other", {}).get("n", 0), ST[n]["entities"])
    short_set = lambda n: pct(sum(e["short"] for e in ST[n]["entity_types"].values()), ST[n]["entities"])
    tz = [f"gemini/itdoc_{c}" for c in "tuvwxyz"]
    uz = tz[1:]
    nz = [f"gemini/itdoc_{c}" for c in "nopqrstuvwxyz"]
    skel = lambda ns: sum(ST[n]["duplicates"]["distinct_word_patterns"] for n in ns)
    val = lambda n, t, k: ST[n]["validity"].get(t, {}).get(k)
    dev = {r: [n for n in sets_of(r) if ST[n]["role"] == "dev"] for r in rn}
    test = {r: [n for n in sets_of(r) if ST[n]["role"] == "test"] for r in rn}
    tpl = lambda n: set(ST[n]["tpl_x_lang"])
    spark_nested = all(tpl(a) <= tpl(b) for a, b in zip(sets_of("spark"), sets_of("spark")[1:]))
    spark_dev_tpl = set().union(*(tpl(n) for n in dev["spark"]))
    spark_new = {n: len(tpl(n) - spark_dev_tpl) for n in test["spark"]}
    low = lambda n, t, k: val(n, t, k) is not None and val(n, t, k) < 50
    bad_cin = [n for n in sets_of("gemini") if low(n, "iban", "fully_valid_pct")]
    bad_cf = [n for n in sets_of("gemini") if low(n, "cf", "fully_valid_pct")]
    bad_cup = [n for n in sets_of("gemini") if low(n, "cup", "format_pct")]
    lt = lambda ns: "".join(n[-1] for n in ns)
    lowsp = sum(v for k, v in G["small_print_px"].items() if k != "None" and int(k) < 11)
    declared_sp = sum(v for k, v in G["small_print_px"].items() if k != "None")
    sp_sets = [n for n in sets_of("gemini") if any(k != "None" and int(k) < 11 for k in ST[n]["small_print_px"])]
    bbox_sets = [n for n in sets_of("gemini") if ST[n]["bbox_most_common"] and ST[n]["bbox_most_common"][0] == [60, 60, 100, 100]]
    bbox_placeholder = sum(ST[n]["bbox_most_common"][1] for n in bbox_sets)
    sent2 = lambda r: [ST[n]["sentences"]["sentences_on_more_than_2pct_of_pages"] for n in sets_of(r)]
    maxs = lambda r: max(ST[n]["sentences"]["max_pages_one_sentence_pct"] for n in sets_of(r))
    sim = lambda n: ST[n]["similarity"]["same_template"]["word_ratio_median"]
    simo = lambda n: ST[n]["similarity"]["other_template"]["word_ratio_median"]
    rng = lambda xs, f=fd: f"{f(min(xs))}–{f(max(xs))}"
    L = G["language"]
    en_ns = sum(ST[n]["language"]["en_pages_italian_stopwords_ge_20pct"] for n in nz[:6])
    tied = [n for n in SETS if ST[n]["tpl_x_level_empty_cells"] and not n.startswith("chatgpt")]
    tot = D["totals"]
    gb = lambda b: fd(b / 1e9, 1) + " GB"
    stray = {n: d["files_not_listed_in_meta"] for n, d in D["sets"].items() if d["files_not_listed_in_meta"]}
    lk = X["entity_strings_eval_also_in_train"]
    sl = X["sentences_eval_also_in_train"]
    ovs = X["sentence_pattern_overlap_between_sets"]
    eld = X["entity_strings_test_also_in_dev_same_renderer"]
    P = []
    A_ = P.append

    # ------------------------------------------------------------ page 1
    A_(f"<h1>Audit dei dataset sintetici OCR — testi e metadati</h1><div class='sub'>14 ottobre 2026 · 63 set, {fi(sum(RR[r]['pages'] for r in rn))} pagine · "
       f"solo <code>meta.json</code> e <code>.txt</code>: nessun PNG è stato aperto · nessun modello eseguito · numeri in <code>work/dataset_audit/stats.json</code></div>")
    A_(f"""<div class="verdict"><p><b>Verdetto.</b> La collezione è congelata per testi e metadati ({fi(tot['files_hashed'])} file, {fi(tot['bytes_hashed'])} byte, manifest SHA-256);
le immagini <b>non</b> sono ancora coperte ({fi(tot['png_files_NOT_hashed'])} PNG, {gb(tot['png_bytes_NOT_hashed'])}).</p>
<p><b>spark</b> è l'unico renderer che rispetta il contratto del brief sulle entità: 0 entità assenti dal testo, {fp(K['entity_totals']['ambiguous_pct'], 3)} ambigue, IBAN/partita IVA/codice fiscale validi al 100%.
Ha però un italiano senza accenti e set di test che ripetono template e frasi dei set di sviluppo.</p>
<p><b>gemini</b> (addestramento) è fatto di almeno cinque generazioni diverse sotto lo stesso nome: {fp(G['entity_totals']['ambiguous_pct'])} di entità ambigue,
{fi(G['types_outside_the_brief']['types'])} tipi di entità fuori dal brief, codici non validi nei set {lt(bad_cin)}, pagine «inglesi» con etichette italiane, e nei set u–z
{fi(len(uz) * 1500)} pagine che si riducono a {fi(skel(uz))} scheletri di testo. Le 39.000 pagine valgono molto meno del loro numero.</p>
<p><b>chatgpt</b> resta un prototipo: 24 pagine per set, lingua legata al livello, IBAN senza lettera CIN e codici fiscali non validi ({fp(H['validity']['cf']['fully_valid_pct'])} validi).
Serve per pagine dense (≈{fd(H['entities_per_page']['mean'], 0)} entità), non per decidere.</p>
<p><b>Conseguenza pratica.</b> Il test sui check digit si può valutare solo su spark e sui set gemini a–o. I confronti «inglese contro italiano» su gemini e i confronti per livello su spark e chatgpt non misurano quello che dichiarano.</p></div>""")
    A_("<h3>Difetti, dal più grave</h3><ol class='def'>")
    D_ = lambda k, t: A_(f"<li>{sev(k)}{t}</li>")
    D_(1, f"<b>gemini t–z: tipi di entità liberi e pagine quasi identiche.</b> Nei set {lt(tz)} dal {fp(min(other_set(n) for n in tz), 0)} al 100% delle entità ha un tipo che il brief non prevede "
          f"({fi(G['types_outside_the_brief']['types'])} tipi in tutto, {fi(G['types_outside_the_brief']['types_in_one_set_only'])} presenti in un solo set, {fi(G['types_outside_the_brief']['count_of_counts'].get('125', 0))} con esattamente 125 occorrenze); "
          f"entità ambigue {rng([amb_set(n) for n in tz], fp)}. I set u–z hanno {rng([ST[n]['duplicates']['distinct_word_patterns'] for n in uz], fi)} scheletri di parole distinti su 1.500 pagine ciascuno "
          f"({fi(skel(uz))} su {fi(1500 * len(uz))}); due pagine dello stesso template coincidono per {rng([sim(n) for n in tz])} della sequenza di parole (spark: {rng([sim(n) for n in sets_of('spark')])}).")
    D_(1, f"<b>Codici non validi: il test sui check digit non è valutabile ovunque.</b> chatgpt: IBAN con layout italiano {fp(H['validity']['iban']['italian_layout_with_cin_letter_pct'], 0)} (manca la lettera CIN; il mod-97 passa al {fp(H['validity']['iban']['mod97_valid_pct'], 0)}), "
          f"codice fiscale valido {fp(H['validity']['cf']['fully_valid_pct'])} (carattere di controllo giusto {fp(H['validity']['cf']['check_char_valid_pct'])}, lettera del mese valida {fp(H['validity']['cf']['month_letter_valid_pct'])}). "
          f"gemini: CIN dell'IBAN sbagliato nei set {lt(bad_cin)} ({rng([val(n, 'iban', 'fully_valid_pct') for n in bad_cin], fp)} validi), codice fiscale valido solo al {rng([val(n, 'cf', 'fully_valid_pct') for n in bad_cf], fp)} nei set {lt(bad_cf)}, "
          f"CUP fuori formato al 100% nei set {lt(bad_cup)}; in totale gemini: IBAN {fp(G['validity']['iban']['fully_valid_pct'])}, CF {fp(G['validity']['cf']['fully_valid_pct'])}. spark: 100% su tutti i codici.")
    D_(1, f"<b>gemini è eterogeneo: lo stesso nome copre generazioni diverse.</b> {fi(G['pages_missing_meta_keys'].get('small_print_px', 0))} pagine (set n–z) senza <code>gt</code>, <code>small_print_px</code>, <code>overlays</code> e con l'id in un altro ordine "
          f"(<code>_L1_en_template</code>); {fi(lowsp)} pagine su {fi(declared_sp)} che lo dichiarano hanno <code>small_print_px</code> sotto 11 (fino a 6 px, set {lt(sp_sets)}); "
          f"{fi(bbox_placeholder)} entità con <code>bbox</code> segnaposto [60,60,100,100] (set {lt(bbox_sets)}); {fi(G['pages_with_nonstandard_degradation_keys'])} pagine con chiavi di degradazione diverse, "
          f"{fi(G['pages_missing_degradation_params']['noise_sigma'])} senza <code>noise_sigma</code>.")
    D_(2, f"<b>Entità ambigue in gemini: {fp(G['entity_totals']['ambiguous_pct'])}</b> ({fi(G['entity_totals']['ambiguous'])} su {fi(G['entity_totals']['n'])}) compaiono più di una volta nel testo della pagina, contro la regola 6 del brief; "
          f"{fi(G['pages_with_ambiguous_entity'])} pagine su {fi(G['pages'])} ne hanno almeno una. Un blocco saltato non si vede se la stessa stringa è altrove. Per tipo: date {fp(et('gemini', 'date')['ambiguous_pct'])}, contract {fp(et('gemini', 'contract')['ambiguous_pct'])}, "
          f"amount {fp(et('gemini', 'amount')['ambiguous_pct'])}, tipi fuori brief {fp(et('gemini', 'other')['ambiguous_pct'])}. spark {fp(K['entity_totals']['ambiguous_pct'], 3)}, chatgpt {fp(H['entity_totals']['ambiguous_pct'])} (tutti i percent).")
    D_(2, f"<b>Lingua in gemini: le pagine «en» non sono inglesi.</b> Su {fi(L['en_pages'])} pagine etichettate <code>en</code>, {fi(L['en_pages_italian_stopwords_ge_20pct'])} ({fp(pct(L['en_pages_italian_stopwords_ge_20pct'], L['en_pages']))}) hanno almeno il 20% di parole funzionali italiane, "
          f"{fi(L['en_pages_italian_stopwords_ge_50pct'])} ({fp(pct(L['en_pages_italian_stopwords_ge_50pct'], L['en_pages']))}) almeno il 50%; {fi(L['en_pages_italian_labels_ge_50pct'])} ({fp(pct(L['en_pages_italian_labels_ge_50pct'], L['en_pages']))}) hanno più etichette italiane che inglesi. "
          f"Nei set n–s: {fi(en_ns)} su 1.800. spark: {fi(K['language']['en_pages_italian_stopwords_ge_20pct'])} su {fi(K['language']['en_pages'])}; chatgpt: 0.")
    D_(2, f"<b>spark scrive l'italiano senza accenti.</b> {fi(K['unicode']['apostrophe_as_accent_words'])} parole con apostrofo al posto dell'accento (sara’, dovra’, proprieta’) su {fi(K['unicode']['it_pages_with_apostrophe_as_accent'])} delle {fi(K['language']['it_pages'])} pagine italiane "
          f"({fp(pct(K['unicode']['it_pages_with_apostrophe_as_accent'], K['language']['it_pages']))}); lettere accentate in tutto il renderer: {fi(sum(c for u, nm, c in K['non_ascii_chars'] if 'LETTER' in nm))}. Il brief chiedeva accenti corretti. "
          f"Su spark non si misura il recupero delle parole accentate, e un modello che corregge in «sarà» viene penalizzato sul CER. Inoltre trattino lungo (—) su tutte le {fi(K['pages'])} pagine e apostrofo tipografico (’) su {fi(K['unicode']['pages_with_char_class']['curly_single'])}.")
    D_(2, f"<b>In spark i set di test non sono indipendenti da quelli di sviluppo.</b> Ogni set contiene i template del precedente più quattro ({'verificato' if spark_nested else 'NON verificato'}: {ST['spark/itdoc_a']['templates']} in a, {ST['spark/itdoc_k']['templates']} in k); "
          f"template nuovi rispetto ai set dev: {', '.join(f'{n[-1]} {v}/{ST[n][chr(116) + chr(101) + chr(109) + chr(112) + chr(108) + chr(97) + chr(116) + chr(101) + chr(115)]}' for n, v in spark_new.items())}. Dal {fp(min(v['pattern_occurrences_also_in_dev_pct'] for v in ovs['spark']['test_sets_vs_dev_sets'].values()), 0)} "
          f"all'{fp(max(v['pattern_occurrences_also_in_dev_pct'] for v in ovs['spark']['test_sets_vs_dev_sets'].values()), 0)} delle frasi di un set di test ha lo stesso schema di una frase dev; le PEC sono le stesse al 100% ({fi(K['entity_repetition']['pec']['distinct'])} indirizzi in tutto il renderer). "
          f"Un risultato su spark test dice come va su spark, non su documenti nuovi.")
    D_(2, f"<b>I livelli non sono ordinati (dai parametri dichiarati).</b> spark: L5 ha meno sfocatura di L4 (mediana {fd(K['degradation']['blur']['5']['p50'])} contro {fd(K['degradation']['blur']['4']['p50'])}) e meno rotazione ({fd(K['degradation']['rotation_deg']['5']['p50'])}° contro {fd(K['degradation']['rotation_deg']['4']['p50'])}°); "
          f"il rumore di L4 e L5 si sovrappone ({fp(K['degradation']['noise_L4_pages_at_or_above_L5_minimum_pct'], 0)} delle pagine L4 sopra il minimo di L5). chatgpt: L3, L4 e L5 hanno la stessa sfocatura ({fd(H['degradation']['blur']['3']['p50'])}, {fd(H['degradation']['blur']['4']['p50'])}, {fd(H['degradation']['blur']['5']['p50'])}) "
          f"e lo stesso rumore ({fd(H['degradation']['noise_sigma']['3']['p50'], 1)}, {fd(H['degradation']['noise_sigma']['4']['p50'], 1)}, {fd(H['degradation']['noise_sigma']['5']['p50'], 1)}); cambia solo la qualità JPEG. gemini è ordinato. Verifica sui pixel non fatta.")
    D_(3, f"<b>Testo ripetuto oltre la soglia del 2% del brief.</b> Segmenti identici (≥30 caratteri) su più del 2% delle pagine del set: gemini {rng(sent2('gemini'), fi)} per set (massimo {fp(maxs('gemini'), 0)} delle pagine), spark {rng(sent2('spark'), fi)} (massimo {fp(maxs('spark'), 0)}), "
          f"chatgpt: un segmento sull'{fp(H['sentences']['max_pages_one_sentence_pct'], 0)} delle 624 pagine, {fi(H['sentences']['distinct_word_patterns'])} schemi di frase in tutto il renderer, {fi(H['companies']['distinct'])} ragioni sociali.")
    D_(3, f"<b>Posizione delle entità sbilanciata.</b> spark: partita IVA {fp(et('spark', 'vat')['last3_pct'], 0)}, codice fiscale {fp(et('spark', 'cf')['last3_pct'], 0)}, IBAN {fp(et('spark', 'iban')['last3_pct'], 0)} nelle ultime tre righe: il recall di questi tipi è il recall del piè di pagina. "
          f"chatgpt: {fp(H['entity_totals']['first3_pct'])} nelle prime tre righe e {fp(H['entity_totals']['last3_pct'])} nelle ultime tre: non misura intestazioni e piè di pagina saltati.")
    D_(3, f"<b>Entità «gratuite».</b> Leggendo il testo della pagina successiva dello stesso set si trova comunque il {fp(G['entity_totals']['in_next_page_pct'])} delle entità in gemini (percent {fp(et('gemini', 'percent')['in_next_page_pct'], 0)}, amount {fp(et('gemini', 'amount')['in_next_page_pct'], 0)}; set f {fp(ST['gemini/itdoc_f']['wrong_page_recall_pct'], 0)}), "
          f"{fp(H['entity_totals']['in_next_page_pct'])} in chatgpt (percent {fp(et('chatgpt', 'percent')['in_next_page_pct'], 0)}: tre stringhe in tutto), {fp(K['entity_totals']['in_next_page_pct'])} in spark. Entità sotto 5 caratteri: gemini {fp(G['entity_totals']['short_pct'])}, spark {fp(K['entity_totals']['short_pct'])}, chatgpt {fp(H['entity_totals']['short_pct'])}.")
    D_(3, f"<b>Template legato al livello in {', '.join(tied)}</b> ({', '.join(str(ST[n]['tpl_x_level_empty_cells']) + ' celle vuote su ' + str(ST[n]['templates'] * 5) for n in tied)}): in questi set il confronto per livello è un confronto tra template. "
          f"chatgpt: lingua legata al livello (noto), 2 pagine per template.")
    D_(4, f"<b>Formato del testo di riferimento.</b> Celle unite con « | » su una riga in {fi(G['unicode']['pages_with_pipe_joined_cells'])} pagine gemini ({fp(pct(G['unicode']['pages_with_pipe_joined_cells'], G['pages']), 0)}) e in tutte le chatgpt (il brief: una riga per cella); "
          f"in gemini {fi(G['unicode']['unaccented_words'])} parole senza accento (societa, qualita…) contro {fi(G['unicode']['accented_words'])} scritte bene, ideogrammi CJK in {fi(sum(c for u, nm, c in G['non_ascii_chars'] if 'CJK' in nm))} occorrenze; "
          f"notazione dei percent sbagliata per lingua in gemini (pagine it {fp(G['validity']['percent']['it_pages_in_english_notation_pct'])}, pagine en {fp(G['validity']['percent']['en_pages_in_italian_notation_pct'])}); "
          f"{sum(len(v) for v in stray.values())} file fuori da <code>meta.json</code> in {', '.join(stray)}.")
    A_("</ol>")
    A_(f"<p class='note'>Cosa è in ordine: nessuna entità assente dal testo (0 su {fi(sum(RR[r]['entities'] for r in rn))}); nessuna entità elencata due volte nella stessa pagina; nessun <code>.txt</code> identico a un altro; tutto NFC; nessuna sequenza di escape letterale; "
       f"nessuna pagina ruotata di 90°; seed unici e secondo la regola SEED×1000+indice; gemini e spark bilanciati 240/60 per lingua a ogni livello; nessuna pagina a rischio di superare i 3.000 token (massimo {fi(max(RR[r]['chars']['max'] for r in rn))} caratteri).</p>")

    # ------------------------------------------------------------ 1 freeze
    A_("<h2 class='pb'>1. Cosa è stato congelato</h2>")
    A_(f"<p>SHA-256 di ogni <code>meta.json</code> e di ogni <code>.txt</code> presenti nelle 63 cartelle (anche i <code>.txt</code> non elencati in <code>meta.json</code>). <b>Le immagini non sono coperte</b>: i PNG sono stati solo contati con <code>os.stat</code>, senza leggerli. "
       f"Verifica: <code>cd {esc(D['root'])} &amp;&amp; shasum -a 256 -c manifest.sha256</code>. SHA-256 del manifest: <code>{D['manifest_sha256']}</code>. Il digest di un set è lo SHA-256 delle righe ordinate «hash, due spazi, nome file».</p>")
    rows = []
    for r in rn:
        ds = [D["sets"][n] for n in sets_of(r)]
        rows.append([r, len(ds), fi(sum(d["pages_in_meta"] for d in ds)), fi(sum(d["files_hashed"] for d in ds)), fi(sum(d["bytes_hashed"] for d in ds)),
                     fi(sum(d["png_files_NOT_hashed"] for d in ds)), gb(sum(d["png_bytes_NOT_hashed"] for d in ds))])
    rows.append(["<b>totale</b>", 63, fi(sum(d["pages_in_meta"] for d in D["sets"].values())), fi(tot["files_hashed"]), fi(tot["bytes_hashed"]), fi(tot["png_files_NOT_hashed"]), gb(tot["png_bytes_NOT_hashed"])])
    A_(tbl(["Renderer", "Set", "Pagine in meta.json", "File con hash (txt + meta)", "Byte con hash", "PNG <b>senza</b> hash", "Byte PNG senza hash"], rows))
    if stray:
        A_("<p>File presenti ma non elencati in <code>meta.json</code>: " + "; ".join(f"<code>{n}</code>: {sum(x.endswith('.txt') for x in v)} .txt (nel manifest) e {sum(x.endswith('.png') for x in v)} .png" for n, v in stray.items()) + ". Il digest per set è nell'appendice.</p>")

    # ------------------------------------------------------------ 2 structure
    A_("<h2>2. Struttura e bilanciamento</h2>")
    rows = []
    for r in rn:
        v = RR[r]
        rows.append([r, v["sets"], fi(v["pages"]), f"{fi(v['lang'].get('it', 0))} / {fi(v['lang'].get('en', 0))}", " / ".join(fi(v["level"].get(str(l), 0)) for l in range(1, 6)),
                     v["template_names"]["distinct"], v["template_names"]["sum_over_sets"], len(v["fonts"]), v["rot90"], fi(v["pages_with_overlays"]),
                     ", ".join(f"{k}:{fi(n)}" for k, n in v["small_print_px"].items() if k != "None") + (f"; assente:{fi(v['small_print_px']['None'])}" if "None" in v["small_print_px"] else "")])
    A_(tbl(["Renderer", "Set", "Pagine", "it / en", "L1 / L2 / L3 / L4 / L5", "Nomi di template distinti", "Somma sui set", "Font", "rot90", "Pagine con overlay", "small_print_px : pagine"], rows))
    A_(f"<p><b>Lingua × livello.</b> gemini e spark: 240 pagine italiane e 60 inglesi a ogni livello in ogni set. chatgpt: {H['lang_x_level'].get('en', {})} per l'inglese e {H['lang_x_level'].get('it', {})} per l'italiano: il livello 5 è solo inglese, e la proporzione è {fp(pct(H['lang']['it'], H['pages']))} / {fp(pct(H['lang']['en'], H['pages']))}, non 80/20. "
       f"<b>Template × lingua</b>: nessuna cella vuota in gemini e spark; in chatgpt {ST['chatgpt/itdoc_a']['tpl_x_lang_empty_cells']} template su 12 per set non hanno pagine inglesi. "
       f"<b>Template × livello</b>: gemini 25 pagine per cella ovunque; spark da {min(ST[n]['tpl_x_level_min_max_cell'][0] for n in sets_of('spark'))} a {max(ST[n]['tpl_x_level_min_max_cell'][1] for n in sets_of('spark'))} pagine per cella, con celle vuote in {', '.join(tied)}; "
       f"le pagine per template in spark scendono da {ST['spark/itdoc_a']['pages_per_template_min_max'][0]}–{ST['spark/itdoc_a']['pages_per_template_min_max'][1]} (a) a {ST['spark/itdoc_k']['pages_per_template_min_max'][0]}–{ST['spark/itdoc_k']['pages_per_template_min_max'][1]} (k).</p>")
    A_(f"<p><b>Seed.</b> Unici in ogni set e uguali a SEED×1000+indice in tutti i 63 set; nessun valore usato da due set dello stesso renderer. Tra renderer i seed coincidono per costruzione (gemini–spark {fi(X['seed_values_shared_between_renderers']['gemini~spark'])}, gemini–chatgpt {fi(X['seed_values_shared_between_renderers']['gemini~chatgpt'])}): "
       f"il contenuto non coincide (sezione 5). <b>Font</b>: gemini {len(G['fonts'])} famiglie ({fi(min(G['fonts'].values()))}–{fi(max(G['fonts'].values()))} pagine ciascuna), spark {len(K['fonts'])} ({fi(min(K['fonts'].values()))}–{fi(max(K['fonts'].values()))}), chatgpt {len(H['fonts'])} ({min(H['fonts'].values())}–{max(H['fonts'].values())}); "
       f"i nomi non sono confrontabili tra renderer (<code>NotoSans-Regular</code>, <code>Noto Sans</code>, <code>NotoSans</code>).</p>")
    A_("<h3>Template: nomi e contenuto</h3>")
    ts = lambda r, k: RR[r]["template_skeletons"][k]
    A_(f"<p>gemini: <b>{G['template_names']['distinct']} nomi distinti</b>, non 312: sei nomi ({', '.join(G['template_names']['shared'])}) sono condivisi dai set {''.join(sorted(set(''.join(G['template_names']['shared'].values()))))} (312 − 6×7 = 270). "
       f"Il resoconto del renderer (<code>README_DATASET_REPORT.md</code>, 18 set) dichiara 216 = 12×18; sugli stessi 18 set i nomi distinti sono 174. "
       f"Stesso nome non vuol dire stessa pagina: confrontando gli inizi di riga ricorrenti (pagine italiane), due set gemini con lo stesso nome di template hanno indice di Jaccard mediano {fd(ts('gemini', 'same_name_different_set_jaccard')['p50'])} "
       f"({ts('gemini', 'same_name_pairs_disjoint_core')} coppie su {ts('gemini', 'same_name_different_set_jaccard')['n']} senza nulla in comune; fanno eccezione a e b, fino a {fd(ts('gemini', 'same_name_different_set_jaccard')['max'])}). "
       f"In spark è il contrario: stesso nome, stesso testo fisso (mediana {fd(ts('spark', 'same_name_different_set_jaccard')['p50'])}, {ts('spark', 'same_name_pairs_identical_core')} coppie identiche su {fi(ts('spark', 'same_name_different_set_jaccard')['n'])}; nomi diversi {fd(ts('spark', 'different_name_different_set_jaccard')['p50'])}). "
       f"In chatgpt i nomi sono {H['template_names']['distinct']} ma dentro un set template diversi condividono il {fp(100 * ts('chatgpt', 'different_name_same_set_jaccard')['p50'], 0)} dello scheletro: sono etichette su una sola costruzione. La geometria della pagina non è stata verificata (immagini non aperte).</p>")
    A_("<h3>Degradazione dichiarata per livello</h3>")
    dser = lambda k: [(r, COL[r], [(l, RR[r]["degradation"][k][str(l)]["p50"]) for l in range(1, 6) if RR[r]["degradation"][k][str(l)].get("n")]) for r in rn]
    dband = lambda k: {r: [(l, RR[r]["degradation"][k][str(l)]["min"], RR[r]["degradation"][k][str(l)]["max"]) for l in range(1, 6) if RR[r]["degradation"][k][str(l)].get("n")] for r in rn}
    A_('<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px">')
    for k, lab in (("noise_sigma", "noise_sigma (mediana)"), ("blur", "blur (mediana)"), ("rotation_deg", "|rotation_deg| (mediana, gradi)"), ("jpeg_quality", "jpeg_quality (mediana)")):
        A_(fig(line_chart(dser(k), "livello", lab, w=380, h=210, xticks=[1, 2, 3, 4, 5], band=dband(k)), f"<b>{k}</b> per livello. Punto: mediana; fascia: minimo–massimo."))
    A_("</div>")
    A_(f"<p class='note'>Popolazione: tutte le pagine con il parametro in <code>meta.json</code> (gemini {fi(G['pages'] - G['pages_missing_degradation_params']['noise_sigma'])}, spark {fi(K['pages'])}, chatgpt {fi(H['pages'])}). Sono i valori dichiarati dal generatore, non misure sui pixel. "
       f"spark dichiara anche <code>contrast</code>, <code>downsample</code>, <code>perspective_px</code>, <code>fold_shadow</code>, <code>show_through</code>, qui non analizzati. In gemini {G['degradation']['L1_pages_with_any_degradation']} pagine L1 hanno una degradazione dichiarata (rotazione fino a {fd(G['degradation']['rotation_deg']['1']['max'])}°) e L5 arriva a {fd(G['degradation']['rotation_deg']['5']['max'])}° (il brief: 1,8°).</p>")

    A_("<h3>gemini set per set</h3>")
    rows = []
    for n in sets_of("gemini"):
        v = ST[n]
        sp = [int(k) for k in v["small_print_px"] if k != "None"]
        rows.append([n[-1], v["distinct_entity_types"], fp(other_set(n), 0), fp(amb_set(n)), fp(v["wrong_page_recall_pct"]), f"{min(sp)}–{max(sp)}" if sp else "<span class='bad'>assente</span>",
                     "sì" if v["id_fields_in_other_order_than_brief"] else "", fp(val(n, "iban", "fully_valid_pct"), 0), fp(val(n, "cf", "fully_valid_pct"), 0), fp(val(n, "cup", "format_pct"), 0),
                     f"{v['language']['en_pages_italian_stopwords_ge_20pct']}/300", fi(v["duplicates"]["distinct_word_patterns"]), fd(sim(n)), fi(v["sentences"]["distinct_word_patterns"])])
    A_(tbl(["Set", "Tipi", "Entità fuori brief", "Ambigue", "Trovate nella pagina sbagliata", "small_print_px", "Id in altro ordine", "IBAN validi", "CF validi", "CUP in formato", "Pagine en con ≥20% italiano", "Scheletri distinti /1500", "Somiglianza stesso template", "Schemi di frase"], rows, cls="tight"))
    A_("<p class='note'>«–» = il set non ha entità di quel tipo. Scheletro: la pagina ridotta alle sole parole in minuscolo (nomi, numeri e codici tolti). Somiglianza: mediana del rapporto difflib sulle sequenze di parole di 200 coppie di pagine con lo stesso template e la stessa lingua.</p>")

    # ------------------------------------------------------------ 3 text
    A_("<h2 class='pb'>3. Testo</h2>")
    hist = lambda key, r: [(int(k), 100 * v / RR[r]["pages"]) for k, v in RR[r][key].items()]
    A_(fig(line_chart([(r, COL[r], hist("chars_hist_250", r)) for r in rn], "caratteri per pagina (classi di 250, estremo inferiore)", "% delle pagine del renderer", h=250),
           f"<b>Lunghezza del testo di riferimento.</b> Popolazione: tutte le pagine (gemini {fi(G['pages'])}, spark {fi(K['pages'])}, chatgpt {fi(H['pages'])})."))
    rows = []
    for r in rn:
        v = RR[r]
        rows.append([r] + [fi(v["chars"][k]) for k in ("min", "p05", "p50", "p95", "max")] + [fi(v["words"]["p50"]), fi(v["lines"]["p50"]), fi(v["lines"]["max"]),
                    fi(v["token_budget"]["est_tokens_at_2.5_chars"]["max"]), fi(round(v["chars"]["max"] / 2)), v["token_budget"]["pages_over_at_2.0"], v["token_budget"]["pages_over_at_1.5"], v["pages_under_800_chars"]])
    A_(tbl(["Renderer", "Caratteri min", "p5", "mediana", "p95", "max", "Parole (mediana)", "Righe (mediana)", "Righe max", "Token stimati max (2,5 car/token)", "(2,0 car/token)", "Pagine oltre 3.000 a 2,0", "a 1,5", "Pagine sotto 800 caratteri"], rows))
    A_(f"<p><b>Budget di 3.000 token.</b> Nessuna pagina lo supera né a 2,5 né a 2,0 caratteri per token; la più lunga è chatgpt con {fi(H['chars']['max'])} caratteri (≈{fi(round(H['chars']['max'] / 2))} token a 2,0). Solo con 1,5 caratteri per token {H['token_budget']['pages_over_at_1.5']} pagine chatgpt lo supererebbero. "
       f"Il rischio non viene dalla lunghezza del riferimento: resta quello del markup che il modello aggiunge (tabelle) e dei loop, che qui non si misura. Nessuna pagina vuota o quasi vuota. "
       f"La lunghezza non dipende dal livello (mediane L1→L5: gemini {fi(G['chars_by_level']['1']['p50'])}→{fi(G['chars_by_level']['5']['p50'])}, spark {fi(K['chars_by_level']['1']['p50'])}→{fi(K['chars_by_level']['5']['p50'])}); in spark le pagine inglesi sono più corte ({fi(K['chars_by_lang']['en']['p50'])} contro {fi(K['chars_by_lang']['it']['p50'])}).</p>")
    A_("<h3>Lingua dichiarata e lingua del testo</h3>")
    rows = []
    for r in rn:
        l = RR[r]["language"]
        rows.append([r, fi(l["en_pages"]), fi(l["en_pages_italian_stopwords_ge_20pct"]), fi(l["en_pages_italian_stopwords_ge_50pct"]), fi(l["en_pages_italian_labels_ge_50pct"]), fp(l["en_pages_italian_stopword_share"]["p50"]),
                     fi(l["it_pages"]), fi(l["it_pages_english_stopwords_ge_20pct"]), fi(l["it_pages_english_stopwords_ge_50pct"]), fi(l["it_pages_english_labels_ge_50pct"])])
    A_(tbl(["Renderer", "Pagine <code>en</code>", "con ≥20% parole funzionali italiane", "con ≥50%", "con più etichette italiane che inglesi", "quota italiana mediana", "Pagine <code>it</code>", "con ≥20% parole funzionali inglesi", "con ≥50%", "con più etichette inglesi che italiane"], rows))
    A_("<p class='note'>Parole funzionali: due liste chiuse (il, della, che, per… / the, of, and, shall…); quota = italiane / (italiane + inglesi). Etichette: due liste di termini da modulo (codice, fiscale, targa, importo… / code, tax, plate, amount…). Tutte le pagine hanno almeno 5 parole funzionali. "
       "In gemini le pagine «en» dei set n–s hanno titoli inglesi ed etichette di campo italiane (TARGA, CODICE FISCALE, PARTITA IVA).</p>")
    A_("<h3>Unicode e caratteri a rischio</h3>")
    cc = lambda r, k: fi(RR[r]["unicode"]["pages_with_char_class"].get(k, 0))
    rows = [[lab] + [f(r) for r in rn] for lab, f in (
        ("Pagine non NFC", lambda r: RR[r]["unicode"]["pages_not_nfc"]),
        ("Pagine che NFKC modifica", lambda r: fi(RR[r]["unicode"]["pages_changed_by_nfkc"])),
        ("Pagine con escape letterali (\\n, \\t, \\u, \\x), entità o tag HTML", lambda r: sum(v for k, v in RR[r]["unicode"]["pages_with_literal_escape_or_markup"].items() if k != "asterisk_run")),
        ("Pagine con sequenze di asterischi (**)", lambda r: fi(RR[r]["unicode"]["pages_with_literal_escape_or_markup"].get("asterisk_run", 0))),
        ("Pagine con trattino lungo —", lambda r: cc(r, "em_dash")), ("Pagine con trattino medio –", lambda r: cc(r, "en_dash")),
        ("Pagine con apostrofo tipografico ’", lambda r: cc(r, "curly_single")), ("Pagine con virgolette tipografiche o «»", lambda r: fi(RR[r]["unicode"]["pages_with_char_class"].get("curly_double", 0) + RR[r]["unicode"]["pages_with_char_class"].get("guillemet", 0))),
        ("Pagine con spazio non separabile o caratteri a larghezza zero", lambda r: fi(RR[r]["unicode"]["pages_with_char_class"].get("nbsp", 0) + RR[r]["unicode"]["pages_with_char_class"].get("zero_width", 0))),
        ("Pagine con punto elenco •", lambda r: cc(r, "bullet")), ("Pagine con €", lambda r: cc(r, "euro")),
        ("Parole con apostrofo al posto dell'accento", lambda r: fi(RR[r]["unicode"]["apostrophe_as_accent_words"])),
        ("Parole senza accento (societa, piu, qualita…) / con accento", lambda r: f"{fi(RR[r]['unicode']['unaccented_words'])} / {fi(RR[r]['unicode']['accented_words'])}"),
        ("Pagine con celle unite da « | »", lambda r: fi(RR[r]["unicode"]["pages_with_pipe_joined_cells"])),
        ("Pagine con righe vuote / doppi spazi / spazi ai bordi", lambda r: f"{fi(RR[r]['unicode']['pages_with_empty_lines'])} / {fi(RR[r]['unicode']['pages_with_double_spaces'])} / {fi(RR[r]['unicode']['pages_with_edge_spaces'])}"),
        ("Entità con caratteri non ASCII", lambda r: f"{fi(RR[r]['entity_totals']['nonascii'])} ({fp(RR[r]['entity_totals']['nonascii_pct'], 0)})"))]
    A_(tbl(["Misura"] + rn, rows))
    A_(f"<p>Le entità con caratteri non ASCII contengono solo €, £ e ° (nessun trattino lungo, apostrofo tipografico o spazio speciale dentro un'entità): il rischio dei caratteri tipografici pesa su CER e blocchi saltati, non sul recall delle entità. "
       f"NFKC modifica {fi(G['unicode']['pages_changed_by_nfkc'])} pagine gemini (³ in m³). Caratteri non ASCII più frequenti — gemini: {', '.join(f'{u} ×{fi(c)}' for u, nm, c in G['non_ascii_chars'][:5])}; spark: {', '.join(f'{u} ×{fi(c)}' for u, nm, c in K['non_ascii_chars'][:5])}; chatgpt: {', '.join(f'{u} ×{fi(c)}' for u, nm, c in H['non_ascii_chars'][:5])}.</p>")
    A_("<h3>Duplicati, quasi duplicati e testo ripetuto</h3>")
    rows = []
    for r in rn:
        v, d, se = RR[r], RR[r]["duplicates"], RR[r]["sentences"]
        simr = [sim(n) for n in sets_of(r)]
        simor = [simo(n) for n in sets_of(r)]
        rows.append([r, fi(v["pages"]), fi(d["distinct_txt"]), fi(d["distinct_after_masking_digits"]), fi(d["distinct_word_patterns"]), rng(simr), rng(simor), fi(se["distinct_sentences"]), fi(se["distinct_word_patterns"]),
                     fi(se["patterns_covering_half_of_pattern_chars"]), fp(se["chars_in_patterns_seen_on_1pct_of_pages_pct"], 0), fi(v["companies"]["distinct"]), fi(v["companies"]["names_covering_half_of_mentions"])])
    A_(tbl(["Renderer", "Pagine", "<code>.txt</code> distinti", "distinti a cifre mascherate", "scheletri di parole distinti", "Somiglianza stesso template (min–max dei set)", "template diversi", "Segmenti distinti ≥30 car.", "Schemi di frase distinti",
            "Schemi che coprono metà del testo", "Testo in schemi presenti su ≥1% delle pagine", "Ragioni sociali distinte", "che coprono metà delle menzioni"], rows, cls="tight"))
    A_(fig(set_bars({n: sim(n) for n in SETS}, "somiglianza mediana (0–1)", "set (lettera), per renderer", ymax=1),
           "<b>Quanto si somigliano due pagine dello stesso template e della stessa lingua.</b> Mediana del rapporto difflib sulle sequenze di parole; 200 coppie casuali per set (seme fisso), 63 set. 1 = stesse parole nello stesso ordine."))
    A_(f"<p>Non ci sono <code>.txt</code> identici, né dentro un set né tra set né tra renderer. Ma in gemini {fi(G['duplicates']['pages_in_digit_masked_groups'])} pagine sono uguali a un'altra a meno delle cifre (gruppi fino a {G['duplicates']['largest_digit_masked_group']} pagine) e gli scheletri di parole distinti sono {fi(G['duplicates']['distinct_word_patterns'])} su {fi(G['pages'])}: "
       f"t ne ha {fi(ST['gemini/itdoc_t']['duplicates']['distinct_word_patterns'])}, u–z {fi(skel(uz))} in tutto. Nei set a–s lo scheletro è distinto per quasi ogni pagina, ma due pagine dello stesso template condividono comunque il {fp(100 * min(sim(n) for n in sets_of('gemini')[:19]), 0)}–{fp(100 * max(sim(n) for n in sets_of('gemini')[:19]), 0)} delle parole. "
       f"Schemi di frase distinti per set: da {fi(max(ST[n]['sentences']['distinct_word_patterns'] for n in sets_of('gemini')))} a {fi(min(ST[n]['sentences']['distinct_word_patterns'] for n in sets_of('gemini')))} (il brief ne chiedeva almeno 300 per lingua); spark {rng([ST[n]['sentences']['distinct_word_patterns'] for n in sets_of('spark')], fi)}; chatgpt {rng([ST[n]['sentences']['distinct_word_patterns'] for n in sets_of('chatgpt')], fi)}.</p>")
    A_("<p>Segmenti più ripetuti (solo set di addestramento e di sviluppo; pagine su cui compaiono nel renderer):</p>")
    A_(tbl(["Renderer", "Pagine", "Segmento"], [[r, fi(c), esc(t[:120])] for r in rn for c, t in RR[r]["top_repeated_sentences_train_dev_only"][:4]], num_from=3))

    # ------------------------------------------------------------ 4 entities
    A_("<h2 class='pb'>4. Entità</h2>")
    A_(fig(line_chart([(r, COL[r], hist("entities_hist_5", r)) for r in rn], "entità per pagina (classi di 5, estremo inferiore)", "% delle pagine del renderer", h=230),
           f"<b>Entità per pagina.</b> Popolazione: tutte le pagine. Medie: gemini {fd(G['entities_per_page']['mean'], 1)}, spark {fd(K['entities_per_page']['mean'], 1)}, chatgpt {fd(H['entities_per_page']['mean'], 1)}; per 1.000 caratteri: {fd(G['entity_density_per_1000_chars'], 1)}, {fd(K['entity_density_per_1000_chars'], 1)}, {fd(H['entity_density_per_1000_chars'], 1)}. Minimo per pagina: {G['entities_per_page']['min']}, {K['entities_per_page']['min']}, {H['entities_per_page']['min']} (il brief: almeno 6)."))
    cats = CANON + ["other"]
    A_(fig(grouped_hbars([c if c != "other" else "fuori brief" for c in cats], [(r, COL[r], [X["type_mix_pct"][r][c] for c in cats]) for r in rn], "% delle entità del renderer"),
           f"<b>Composizione per tipo.</b> Popolazione: tutte le entità (gemini {fi(G['entities'])}, spark {fi(K['entities'])}, chatgpt {fi(H['entities'])})."))
    ob = G["types_outside_the_brief"]
    A_(f"<p><b>Tipi fuori dal brief (solo gemini).</b> {fi(ob['types'])} tipi oltre gli undici previsti, {fi(ob['entities'])} entità ({fp(ob['entities_pct'])}); {fi(ob['types_in_one_set_only'])} tipi esistono in un solo set; "
       f"{fi(ob['count_of_counts'].get('125', 0))} hanno 125 occorrenze, {fi(ob['count_of_counts'].get('250', 0))} ne hanno 250: sono campi di un singolo template (125 pagine). I più frequenti: {', '.join(f'<code>{t}</code> {fi(c)}' for t, c in ob['top'][:8])}. "
       f"Il recall per tipo su questi set non è confrontabile con nessun altro set.</p>")
    A_("<h3>Ambiguità, entità corte, entità trovate per caso</h3>")
    rows = []
    for t in cats:
        row = [t if t != "other" else "fuori brief"]
        for r in rn:
            e = et(r, t)
            row += [fi(e["n"]), fp(e["short_pct"]), fp(e["ambiguous_pct"]), fp(e["sub_pct"]), fp(e["in_next_page_pct"])] if e else ["–"] * 5
        rows.append(row)
    row = ["<b>tutte</b>"]
    for r in rn:
        e = RR[r]["entity_totals"]
        row += [fi(e["n"]), fp(e["short_pct"]), fp(e["ambiguous_pct"]), fp(e["sub_pct"]), fp(e["in_next_page_pct"])]
    rows.append(row)
    A_(tbl(["Tipo"] + [f"{r}: {c}" for r in rn for c in ("n", "&lt;5 car.", "ambigue", "dentro un'altra", "pagina sbagliata")], rows, cls="tight"))
    A_(f"<p class='note'>Ambigua: la stringa compressa (NFKC, minuscole, senza spazi) compare più di una volta nel testo compresso della pagina, oppure è contenuta in un'altra entità della stessa pagina («dentro un'altra»; in questi dati ogni entità contenuta in un'altra compare anche più volte). "
       f"Pagina sbagliata: l'entità si trova nel testo della pagina successiva dello stesso set, cioè il recall che si ottiene leggendo un'altra pagina. Entità assenti dal testo della propria pagina: 0 in tutti i renderer; entità elencate due volte: 0.</p>")
    A_(fig(set_bars({n: amb_set(n) for n in SETS}, "% di entità ambigue", "set (lettera), per renderer"), "<b>Entità ambigue per set.</b> Popolazione: tutte le entità di ogni set, 63 set."))
    A_("<h3>Ripetizione delle stesse stringhe</h3>")
    rows = []
    for t in cats:
        row = [t if t != "other" else "fuori brief"]
        for r in rn:
            e = RR[r]["entity_repetition"].get(t)
            row += [fi(e["distinct"]), fp(e["uniqueness_pct"]), fi(e["max_repeats_one_string"])] if e else ["–"] * 3
        rows.append(row)
    A_(tbl(["Tipo"] + [f"{r}: {c}" for r in rn for c in ("stringhe distinte", "unicità", "ripetizioni max")], rows, cls="tight"))
    A_(f"<p>Unicità = stringhe distinte / occorrenze, sull'intero renderer. Codici (IBAN, CIG, CUP) unici ovunque; partita IVA e codice fiscale unici in spark e chatgpt, quasi in gemini ({fp(G['entity_repetition']['vat']['uniqueness_pct'])} e {fp(G['entity_repetition']['cf']['uniqueness_pct'])}, una stringa fino a {G['entity_repetition']['cf']['max_repeats_one_string']} volte). "
       f"Le PEC sono un insieme chiuso: {fi(K['entity_repetition']['pec']['distinct'])} indirizzi per {fi(K['entity_repetition']['pec']['occurrences'])} occorrenze in spark, {fi(G['entity_repetition']['pec']['distinct'])} per {fi(G['entity_repetition']['pec']['occurrences'])} in gemini. "
       f"I percent sono {K['entity_repetition']['percent']['distinct']} stringhe in spark e {H['entity_repetition']['percent']['distinct']} in chatgpt: non distinguono una pagina dall'altra. In gemini un importo è ripetuto fino a {G['entity_repetition']['amount']['max_repeats_one_string']} volte (unicità {fp(G['entity_repetition']['amount']['uniqueness_pct'])}).</p>")
    A_("<h3>Posizione nel testo</h3>")
    segs = [("prime 3 righe", "#4a3aa7"), ("corpo", "#dcdbd5"), ("ultime 3 righe", "#eda100")]
    rows = []
    for r in rn:
        e = RR[r]["entity_totals"]
        rows.append((f"{r} — tutte", [e["first3_pct"], 100 - e["first3_pct"] - e["last3_pct"], e["last3_pct"]]))
    for t in ("iban", "vat", "cf", "pec", "contract", "amount"):
        for r in rn:
            e = et(r, t)
            rows.append((f"{r} — {t}", [e["first3_pct"], 100 - e["first3_pct"] - e["last3_pct"], e["last3_pct"]]))
    A_(fig(stacked_rows(rows, segs, "% delle entità"), "<b>Dove sta la prima occorrenza di ogni entità nel testo di riferimento.</b> Popolazione: tutte le entità del renderer e, sotto, sei tipi. Righe = righe non vuote del <code>.txt</code> (un blocco di testo per riga)."))
    A_(f"<p>spark concentra i dati societari nel piè di pagina: chi salta il piè perde quasi tutte le partite IVA e i codici fiscali, quindi «recall per tipo» e «recall del piè di pagina» sono la stessa misura. gemini ha il {fp(et('gemini', 'contract')['first3_pct'], 0)} dei <code>contract</code> nelle prime tre righe e il {fp(et('gemini', 'hand')['last3_pct'], 0)} delle <code>hand</code> nelle ultime tre. "
       f"chatgpt ha tutto nel corpo.</p>")
    A_("<h3>Validità formale dei codici</h3>")
    V = lambda r, t, k, d=1: fp(RR[r]["validity"].get(t, {}).get(k), d)
    rows = [[lab] + [f(r) for r in rn] for lab, f in (
        ("IBAN: IT + 25 caratteri", lambda r: V(r, "iban", "shape_IT_plus_25_pct")), ("IBAN: layout italiano con lettera CIN", lambda r: V(r, "iban", "italian_layout_with_cin_letter_pct")),
        ("IBAN: mod-97 valido", lambda r: V(r, "iban", "mod97_valid_pct")), ("IBAN: CIN valido", lambda r: V(r, "iban", "cin_valid_pct")), ("<b>IBAN: tutto valido</b>", lambda r: V(r, "iban", "fully_valid_pct")),
        ("IBAN: scritto a gruppi di quattro", lambda r: V(r, "iban", "written_in_groups_of_4_pct")),
        ("Partita IVA: 11 cifre", lambda r: V(r, "vat", "format_11_digits_pct")), ("<b>Partita IVA: cifra di controllo valida</b>", lambda r: V(r, "vat", "check_digit_valid_pct")),
        ("Codice fiscale: forma a 16 caratteri", lambda r: V(r, "cf", "shape_16_pct")), ("Codice fiscale: lettera del mese valida", lambda r: V(r, "cf", "month_letter_valid_pct")),
        ("Codice fiscale: giorno valido (01–31, 41–71)", lambda r: V(r, "cf", "day_field_valid_pct")), ("Codice fiscale: carattere di controllo valido", lambda r: V(r, "cf", "check_char_valid_pct")),
        ("<b>Codice fiscale: tutto valido</b>", lambda r: V(r, "cf", "fully_valid_pct")),
        ("CIG: 10 caratteri alfanumerici", lambda r: V(r, "cig", "format_10_alnum_pct")), ("CUP: lettera, 2 cifre, lettera, 11 cifre", lambda r: V(r, "cup", "format_pct")),
        ("PEC: indirizzo ben formato / dominio .example", lambda r: f"{V(r, 'pec', 'address_format_pct', 0)} / {V(r, 'pec', 'domain_example_pct', 0)}"),
        ("Date: data di calendario esistente", lambda r: V(r, "date", "calendar_valid_pct")),
        ("Importi su pagine it in notazione inglese", lambda r: V(r, "amount", "it_pages_in_english_notation_pct")), ("Importi su pagine en in notazione italiana", lambda r: V(r, "amount", "en_pages_in_italian_notation_pct")),
        ("Importi con due o più gruppi di migliaia", lambda r: V(r, "amount", "two_or_more_thousands_groups_pct")),
        ("Percent su pagine it in notazione inglese", lambda r: V(r, "percent", "it_pages_in_english_notation_pct")), ("Percent su pagine en in notazione italiana", lambda r: V(r, "percent", "en_pages_in_italian_notation_pct")))]
    A_(tbl(["Controllo (quota delle entità del tipo)"] + [f"{r} (n: iban {fi(et(r, 'iban').get('n', 0))}, vat {fi(et(r, 'vat').get('n', 0))}, cf {fi(et(r, 'cf').get('n', 0))})" for r in rn], rows))
    tl = G["validity"].get("tail_types_named_like_codes", {})
    dts = G["validity"]["date"]["en_pages"]
    A_(f"<p>In chatgpt gli IBAN sono «IT» + 25 cifre: superano il mod-97 ma non hanno la lettera CIN, quindi non sono IBAN italiani; i codici fiscali hanno la forma giusta ma mese, giorno e carattere di controllo a caso. "
       f"In gemini il mod-97 è sempre valido, il CIN no ({fp(G['validity']['iban']['cin_valid_pct'])}): un controllo che usi solo il mod-97 funziona su tutti e tre i renderer, uno che usi il CIN no. "
       f"Tra i tipi fuori brief di gemini il cui nome indica un codice: {fi(tl.get('vat', {}).get('n', 0))} partite IVA con cifra di controllo valida al {fp(tl.get('vat', {}).get('check_digit_valid_pct'))}, {fi(tl.get('iban', {}).get('n', 0))} IBAN (mod-97 {fp(tl.get('iban', {}).get('mod97_valid_pct'), 0)}), {fi(tl.get('cf', {}).get('n', 0))} codici fiscali ({fp(tl.get('cf', {}).get('check_char_valid_pct'), 0)}). "
       f"Nel testo di gemini ci sono {fi(G['iban_like_strings_in_text']['not_listed_as_entity'])} stringhe con forma di IBAN non elencate come entità. Sulle pagine <code>en</code> di gemini {fi(dts.get('it_num', 0))} date su {fi(sum(dts.values()))} sono nel formato italiano gg/mm/aaaa. "
       f"Il CIG reale ha un controllo proprio che il brief non chiedeva: qui si verifica solo la forma. Importi con più gruppi di migliaia (quelli che espongono lo scambio di notazione su più separatori): pochi in gemini e spark, quasi tutti in chatgpt.</p>")

    # ------------------------------------------------------------ 5 leakage
    A_("<h2 class='pb'>5. Perdite tra addestramento e valutazione</h2>")
    rows = []
    for t in CANON + ["ALL"]:
        row = [t if t != "ALL" else "<b>tutte</b>"]
        for r in ("spark", "chatgpt"):
            e = lk[r].get(t)
            row += [fi(e["distinct"]), fi(e["distinct_also_in_train"]), fp(e["occurrences_also_in_train_pct"])] if e else ["–"] * 3
        rows.append(row)
    A_(tbl(["Tipo"] + [f"{r}: {c}" for r in ("spark", "chatgpt") for c in ("stringhe distinte", "anche in gemini", "occorrenze anche in gemini")], rows))
    A_(f"<p><b>Tra gemini (train) e la valutazione non c'è perdita di contenuto.</b> Frasi identiche: spark {fi(sl['spark']['distinct_identical_in_train'])}, chatgpt {fi(sl['chatgpt']['distinct_identical_in_train'])}; schemi di frase in comune: {fi(sl['spark']['distinct_patterns_also_in_train'])} su {fi(sl['spark']['distinct_patterns'])} e {fi(sl['chatgpt']['distinct_patterns_also_in_train'])} su {fi(sl['chatgpt']['distinct_patterns'])}; "
       f"ragioni sociali in comune: {sl['spark']['company_names_also_in_train']} su {fi(sl['spark']['company_names_distinct'])} e {sl['chatgpt']['company_names_also_in_train']} su {sl['chatgpt']['company_names_distinct']}; nessun <code>.txt</code> identico; nessun nome di template in comune tra gemini e spark ({len(X['template_names_shared_between_renderers']['gemini~chatgpt'])} nomi generici tra gemini e chatgpt). "
       f"Le stringhe di entità in comune sono date, percentuali e le date dentro le note a mano, cioè insiemi piccoli per natura. Codici: 0 IBAN, CF, CIG, CUP, PEC; {lk['spark']['vat']['distinct_also_in_train']} partite IVA e {lk['spark']['contract']['distinct_also_in_train']} numeri di contratto di spark esistono anche in gemini "
       f"(per le partite IVA il caso ne darebbe meno di una: i due renderer usano gli stessi seed), su {fi(lk['spark']['vat']['distinct'])} e {fi(lk['spark']['contract']['distinct'])}.</p>")
    A_("<h3>Dentro lo stesso renderer: set di test contro set di sviluppo</h3>")
    rows = []
    for r in ("spark", "chatgpt"):
        for n, v in ovs[r].get("test_sets_vs_dev_sets", {}).items():
            e = eld.get(n, {})
            g = lambda t: fp(e.get(t, {}).get("also_in_dev_sets_pct"), 0)
            rows.append([n, len(tpl(n) - set().union(*(tpl(d) for d in dev[r]))), ST[n]["templates"], fp(v["pattern_occurrences_also_in_dev_pct"], 0), fp(v["distinct_sentences_identical_in_dev_pct"]), g("ALL"), g("pec"), g("date"), g("percent"), g("amount"), g("iban")])
    A_(tbl(["Set di test", "Template non presenti nei set dev", "Template", "Frasi con schema presente nei dev", "Segmenti identici a uno dei dev", "Entità con stringa presente nei dev: tutte", "pec", "date", "percent", "amount", "iban"], rows))
    A_(f"<p>Ruoli da <code>datasets.toml</code> al momento dell'analisi: spark dev = {lt(dev['spark'])}, test = {lt(test['spark'])}; chatgpt test = {lt(test['chatgpt'])}. Lo stesso renderer sta dai due lati e lo fa con gli stessi template, le stesse frasi a slot e lo stesso insieme di PEC. "
       f"In media il {fp(ovs['spark']['mean_pct_of_pattern_occurrences_of_a_set_found_in_another_set'], 0)} delle frasi di un set spark ha lo schema di una frase di un altro set spark (gemini: {fp(ovs['gemini']['mean_pct_of_pattern_occurrences_of_a_set_found_in_another_set'])}; chatgpt: {fp(ovs['chatgpt']['mean_pct_of_pattern_occurrences_of_a_set_found_in_another_set'], 0)}). "
       f"Questo non tocca il confronto train/test (gemini non entra), ma una configurazione scelta sui dev di spark è scelta anche sui suoi test.</p>")

    # ------------------------------------------------------------ 6 comparability
    A_("<h2>6. Confrontabilità dei tre renderer</h2>")
    rows = [[lab] + [f(r) for r in rn] for lab, f in (
        ("Caratteri per pagina (mediana)", lambda r: fi(RR[r]["chars"]["p50"])), ("Righe per pagina (mediana)", lambda r: fi(RR[r]["lines"]["p50"])),
        ("Entità per pagina (media)", lambda r: fd(RR[r]["entities_per_page"]["mean"], 1)), ("Entità per 1.000 caratteri", lambda r: fd(RR[r]["entity_density_per_1000_chars"], 1)),
        ("Quota di importi tra le entità", lambda r: fp(X["type_mix_pct"][r]["amount"], 0)), ("Quota di entità fuori brief", lambda r: fp(X["type_mix_pct"][r]["other"], 0)),
        ("Entità ambigue", lambda r: fp(RR[r]["entity_totals"]["ambiguous_pct"])), ("Entità nelle ultime 3 righe", lambda r: fp(RR[r]["entity_totals"]["last3_pct"])),
        ("small_print_px dichiarato", lambda r: ", ".join(k for k in RR[r]["small_print_px"] if k != "None")), ("Famiglie di font", lambda r: len(RR[r]["fonts"])),
        ("Somiglianza tra pagine dello stesso template", lambda r: rng([sim(n) for n in sets_of(r)])), ("Lettere accentate per 1.000 caratteri", lambda r: fd(1000 * sum(c for u, nm, c in RR[r]["non_ascii_chars"] if "LETTER" in nm) / (RR[r]["chars"]["mean"] * RR[r]["pages"]), 1)),
        ("Pagine con celle unite da « | »", lambda r: fp(pct(RR[r]["unicode"]["pages_with_pipe_joined_cells"], RR[r]["pages"]), 0)))]
    A_(tbl(["Misura"] + rn, rows))
    A_(f"""<p><b>Abbastanza diversi da essere un test indipendente: sì</b>, per contenuto: nessuna frase, schema, ragione sociale o codice in comune tra gemini e gli altri due. <b>Abbastanza simili perché l'addestramento su gemini si trasferisca: non dimostrato dai testi.</b>
gemini ha la metà della densità di entità di spark ({fd(G['entity_density_per_1000_chars'], 1)} contro {fd(K['entity_density_per_1000_chars'], 1)} per 1.000 caratteri) e meno della metà degli importi; un quarto delle sue entità ha tipi che la valutazione non conosce;
scrive le tabelle con celle unite da « | » mentre spark mette una cella per riga; usa gli accenti mentre spark non li ha; dichiara corpi piccoli fino a 6 px mentre spark sta tra 11 e 13 e chatgpt a 14. Sono differenze di convenzione del testo di riferimento, non solo di aspetto: un modello addestrato su gemini impara un formato di uscita diverso da quello su cui viene valutato.</p>""")
    A_(f"""<ul><li><b>gemini — serve per:</b> addestramento, limitatamente ai set a–s (28.500 pagine) e sapendo che due pagine dello stesso template condividono metà o più delle parole. <b>Non usare per:</b> misure per lingua (le pagine «en» sono miste);
recall per tipo (tipi liberi in t–z); test sui check digit nei set {lt(bad_cin)} (CIN) e {lt(bad_cf)} (CF); qualsiasi conteggio che tratti u–z come {fi(1500 * len(uz))} pagine: sono {fi(skel(uz))} scheletri. Le pagine con corpo dichiarato sotto 11 px vanno guardate prima di addestrare.</li>
<li><b>spark — serve per:</b> valutazione del recall delle entità (contratto rispettato, zero ambiguità), test sui check digit, bilanciamento lingua × livello. <b>Non usare per:</b> parole accentate; confronti per livello (L5 non è peggiore di L4 nei parametri);
confronto per livello nei set {', '.join(n[-1] for n in tied)} (template legato al livello); come prova di generalizzazione a documenti nuovi (i test ripetono i template dei dev); il recall per tipo di IVA, CF e IBAN come misura separata dal piè di pagina.</li>
<li><b>chatgpt — serve per:</b> pagine dense con molti importi a più gruppi di migliaia ({fp(H['validity']['amount']['two_or_more_thousands_groups_pct'], 0)}), utili per lo scambio di notazione; controllo rapido. <b>Non usare per:</b> decisioni (24 pagine per set, un solo layout, {H['sentences']['distinct_word_patterns']} schemi di frase);
lingua o livello (legati); check digit di IBAN e codice fiscale; intestazioni e piè di pagina; il tipo percent (tre stringhe, tutte ambigue).</li></ul>""")

    # ------------------------------------------------------------ 7 docs
    A_("<h2>7. Cosa contraddice <code>docs/DATASETS.md</code></h2><ul>")
    for t in (
        f"«All pages … 80% Italian, 20% English; five degradation levels in equal parts»: vero per gemini e spark. chatgpt è {fp(pct(H['lang']['it'], H['pages']))} / {fp(pct(H['lang']['en'], H['pages']))} e i livelli sono {' / '.join(str(H['level'][str(l)]) for l in range(1, 6))}.",
        f"«Short entities: 2.5% (spark, chatgpt) to 7.6% (gemini)»: vale per i primi set. Sui 26 set gemini è {fp(G['entity_totals']['short_pct'])} (da {fp(min(short_set(n) for n in sets_of('gemini')))} a {fp(max(short_set(n) for n in sets_of('gemini')))} per set); spark {fp(K['entity_totals']['short_pct'])} (da {fp(min(short_set(n) for n in sets_of('spark')))} a {fp(max(short_set(n) for n in sets_of('spark')))}).",
        f"«gemini … balanced by language at every level: yes»: vero per l'etichetta <code>lang</code>, non per il testo: {fp(pct(L['en_pages_italian_stopwords_ge_20pct'], L['en_pages']), 0)} delle pagine <code>en</code> ha almeno il 20% di parole funzionali italiane.",
        "«The gemini sets … pass glmocr check … every entity typed»: passano, ma il controllo non guarda né i nomi dei tipi né le chiavi di <code>meta.json</code>: 803 tipi fuori brief e 19.500 pagine senza <code>gt</code>, <code>small_print_px</code>, <code>overlays</code>.",
        "«gemini/itdoc_g to _z … one kind of Italian document per set» con «12 templates each»: i nomi sono 12 per set, ma nei set u–z le 1.500 pagine hanno 24–36 scheletri di testo; il documento non lo dice.",
        f"«Each spark set keeps the templates of the one before and adds four: 14 in a, 54 in k»: confermato. Non detto: di conseguenza i set di test h–k hanno solo {', '.join(str(v) for n, v in spark_new.items() if n != 'spark/itdoc_b')} template che i dev non hanno, e b ne ha {spark_new.get('spark/itdoc_b', '–')}.",
        "«In spark/itdoc_a the levels do not order the pages by difficulty»: coerente con i parametri dichiarati, e vale per tutti gli 11 set spark (L5 con meno sfocatura e rotazione di L4), non solo per a.",
        f"«about 77 entities each», «small print 14 pixels», «48 stray files in chatgpt/itdoc_b», conteggi di entità per set (a 21.358, b 19.083, c 19.875, spark a 37.689, b 37.153): confermati.",
        f"Resoconto di gemini (<code>README_DATASET_REPORT.md</code>): dichiara «altezza maiuscola verificata in [11, 17] px» ({fi(lowsp)} pagine dichiarano meno di 11 nei loro stessi metadati), «ripetizione massima frasi 1,60%» (misurato: un segmento fino al {fp(maxs('gemini'), 0)} delle pagine di un set) e «216 template distinti» per 18 set (174 nomi distinti).",
    ):
        A_(f"<li>{t}</li>")
    A_("</ul>")

    # ------------------------------------------------------------ appendix
    A_("<h2 class='pb'>Appendice A. Tutti i 63 set</h2>")
    rows = []
    for n in SETS:
        v, d = ST[n], D["sets"][n]
        rows.append([n, v["role"] or "–", fi(v["pages"]), v["templates"], fi(v["chars"]["p50"]), fd(v["entities_per_page"]["mean"], 1), fp(short_set(n)), fp(amb_set(n)), fp(v["wrong_page_recall_pct"]),
                     fd(sim(n)), v["sentences"]["sentences_on_more_than_2pct_of_pages"], fi(d["txt_files"]), fi(d["png_files_NOT_hashed"]), fd(d["png_bytes_NOT_hashed"] / 1e9, 2), f"<code>{d['digest'][:16]}</code>"])
    A_(tbl(["Set", "Ruolo", "Pagine", "Template", "Caratteri (mediana)", "Entità/pagina", "&lt;5 car.", "Ambigue", "Pagina sbagliata", "Somiglianza stesso template", "Segmenti su &gt;2% delle pagine", ".txt con hash", "PNG (senza hash)", "PNG GB", "Digest del set (primi 16)"], rows, num_from=2, cls="tight"))
    A_("<p class='note'>Il digest completo di ogni set e lo SHA-256 di ogni <code>meta.json</code> sono in <code>work/dataset_audit/digests.json</code>; tutte le tabelle per set (lingua × livello, template × lingua, template × livello, validità, degradazione) in <code>stats.json</code>.</p>")

    A_("<h2>Appendice B. Definizioni</h2><ul>")
    for t in (
        "<b>Compressione</b> di una stringa: NFKC, minuscole, spazi tolti (<code>metrics.squash</code>): è la forma in cui il progetto cerca le entità.",
        "<b>Segmento</b>: una riga del <code>.txt</code> spezzata dopo . ! ? ; quando segue uno spazio; si contano i segmenti di almeno 30 caratteri compressi, una volta per pagina. <b>Schema di frase</b>: il segmento ridotto alle parole scritte in minuscolo (nomi, numeri e codici tolti), almeno 5 parole.",
        "<b>Scheletro di una pagina</b>: l'intera pagina ridotta allo stesso modo. <b>Cifre mascherate</b>: testo compresso con ogni cifra sostituita da 0.",
        "<b>Scheletro di un template</b> (confronto tra set): gli inizi di riga (prime tre parole, cifre mascherate) presenti su almeno il 30% delle pagine italiane di quel template in quel set.",
        "<b>IBAN</b>: layout IT, 2 cifre, lettera CIN, 10 cifre, 12 caratteri; mod-97 sul codice riordinato; CIN ricalcolato con le tabelle pari/dispari. <b>Partita IVA</b>: 11 cifre e cifra di controllo di Luhn. <b>Codice fiscale</b>: forma, lettera del mese in ABCDEHLMPRST, giorno 01–31 o 41–71, carattere di controllo; omocodie non considerate.",
        "<b>Stima dei token</b>: caratteri del riferimento divisi per 2,5 e per 2,0; il tokenizer non è stato caricato.",
        f"<b>Ruoli</b>: letti da <code>datasets.toml</code> al momento dell'esecuzione ({S['written']}).",
    ):
        A_(f"<li>{t}</li>")
    A_("</ul>")

    # ------------------------------------------------------------ not done
    A_(f"""<h2>Non ancora fatto: immagini</h2>
<p>Per decisione del proprietario questa consegna copre solo testi e metadati. Nessun PNG è stato aperto, decodificato o sottoposto ad hash. Resta da fare, come passo separato dello stesso script (<code>experiments/dataset_audit.py</code>, che non richiede di rifare la parte sui testi):</p>
<ul><li><b>Manifest delle immagini</b>: SHA-256 dei {fi(tot['png_files_NOT_hashed'])} PNG ({gb(tot['png_bytes_NOT_hashed'])}). Finché manca, il congelamento non garantisce che le pagine non cambino.</li>
<li>Dimensione e modalità di ogni PNG (1240×1754, RGB a 8 bit), distribuzione delle dimensioni dei file.</li>
<li>Duplicati di immagine per hash.</li>
<li>Luminosità, contrasto, copertura d'inchiostro, quota di pagina vuota per renderer e template (il brief: «una pagina mezza vuota è una pagina sprecata»).</li>
<li>Degradazione reale: nitidezza, rumore e contrasto misurati a L1 e L5, e confronto con i parametri dichiarati (sezione 2). L'ordine dei livelli qui riportato viene solo da <code>meta.json</code>.</li>
<li>Verifica sui pixel di ciò che i metadati dichiarano: corpo minimo ({fi(lowsp)} pagine gemini sotto 11 px), <code>bbox</code> delle entità, sovrapposizione di timbri, glifi mancanti (ideogrammi CJK in gemini), stesso layout per template con lo stesso nome.</li></ul>
<p><b>Non verificabile nemmeno con le immagini, o non verificato qui:</b> numero reale di token (tokenizer non caricato); se un'entità sta dentro un blocco in corpo piccolo; correttezza dei valori troncati nei moduli gemini; i README dei singoli set oltre al resoconto di portafoglio di gemini; le cartelle <code>huggingface/</code> e <code>kaggle/</code>.</p>""")

    html = f'<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Audit dataset OCR</title><style>{CSS}</style></head><body>{"".join(P)}</body></html>'
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print("report:", OUT / "report.html", len(html), "bytes")
    if pdf:
        pdf = pathlib.Path(pdf).expanduser()
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        subprocess.run([chrome, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", (OUT / "report.html").as_uri()], check=True, capture_output=True, timeout=180)
        n = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
        print("pdf:", pdf, pdf.stat().st_size, "bytes,", n, "pages")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["manifest", "text", "stats", "report", "all"])
    ap.add_argument("--force", action="store_true", help="recompute the per-set cache")
    ap.add_argument("--pdf", help="also print the report to this PDF with headless Chrome")
    a = ap.parse_args()
    if a.step in ("manifest", "all"):
        step_manifest()
    if a.step in ("text", "all"):
        step_text(a.force)
    if a.step in ("stats", "all"):
        step_stats()
    if a.step in ("report", "all"):
        build(a.pdf)


if __name__ == "__main__":
    main()
