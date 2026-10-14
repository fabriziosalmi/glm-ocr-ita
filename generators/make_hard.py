"""Draconian synthetic it/en document set with EXACT ground truth + entities to recover.

python generators/make_hard.py  ->  hard/NN_<tpl>_L<level>_<lang>.png, hard/meta.json
python generators/make_hard.py --out held --seed 20261012 --reps 3 --variants  ->  a second, larger set: other seed, several pages per
    template and level, and layouts that vary (row counts, footnote size and count) instead of being fixed.
Levels: L1 clean | L2 light scan | L3 bad scan, low dpi, handwritten correction | L4 stamp, warp, noise |
        L5 tiny font, heavy noise, low contrast, fold shadow, half the pages rotated 90deg.
The handwritten amount of levels 3 to 5 is an entity but is NOT written in the .txt (kept: the published sets depend on it).
Uses macOS system fonts, so pages regenerate on a Mac only. Do not reorder any rng call: the sets are defined by this exact code.
Metrics: order-sensitive CER on the alnum stream + entity recall (amounts, dates, IBAN, P.IVA, CF, CIG...).
"""
import argparse, json, math, pathlib, random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="hard"); ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--variants", action="store_true")
ARGS = ap.parse_args()
OUT = pathlib.Path(__file__).resolve().parent.parent / ARGS.out; OUT.mkdir(exist_ok=True)
rng = random.Random(ARGS.seed)
def var(default, lo, hi):  # layout knob: fixed unless --variants (then drawn, so the default set stays byte-identical)
    return rng.randint(lo, hi) if ARGS.variants else default
S = "/System/Library/Fonts/"; SS = S + "Supplemental/"
FONTS = {"times": (S+"Times.ttc", 0, 1), "helv": (S+"Helvetica.ttc", 0, 1), "georgia": (SS+"Georgia.ttf", 0, 0),
         "verdana": (SS+"Verdana.ttf", 0, 0), "courier": (SS+"Courier New.ttf", 0, 0), "trebuchet": (SS+"Trebuchet MS.ttf", 0, 0)}
BOLD = {"times": (S+"Times.ttc", 1), "helv": (S+"Helvetica.ttc", 1), "georgia": (SS+"Georgia Bold.ttf", 0),
        "verdana": (SS+"Verdana Bold.ttf", 0), "courier": (SS+"Courier New Bold.ttf", 0), "trebuchet": (SS+"Trebuchet MS Bold.ttf", 0)}
def F(name, size, bold=False):
    p, i = (BOLD if bold else {k: v[:2] for k, v in FONTS.items()})[name][:2]
    return ImageFont.truetype(p, max(8, int(size)), index=i)
HAND = SS + "Bradley Hand Bold.ttf"

# ---------------------------------------------------------------- entities
def amt(lang, n):
    s = f"{n:,.2f}"
    return (s.replace(",", "X").replace(".", ",").replace("X", ".") + " €") if lang == "it" else "$ " + s
def date(lang):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2022, 2026)
    mi = "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split()
    me = "January February March April May June July August September October November December".split()
    return rng.choice([f"{d:02d}/{m:02d}/{y}", f"{d} {mi[m-1]} {y}"]) if lang == "it" else rng.choice([f"{y}-{m:02d}-{d:02d}", f"{me[m-1]} {d}, {y}"])
def iban():
    r = "".join(rng.choice("0123456789") for _ in range(22)); b = f"IT{rng.randint(10,99)}{rng.choice('ABCDEFGHKLMNPQRSTUVWXYZ')}{r}"
    return " ".join(b[i:i+4] for i in range(0, len(b), 4))
def piva(): return "".join(rng.choice("0123456789") for _ in range(11))
def cf(): return "".join(rng.choice("ABCDEFGHLMNPRSTVZ") for _ in range(6)) + f"{rng.randint(50,99)}" + rng.choice("ABCDEHLMPRST") + f"{rng.randint(1,28):02d}" + rng.choice("ABCDEFGHLMZ") + f"{rng.randint(100,999)}" + rng.choice("ABCDEFGHKLMNPRSTUVWXYZ")
def cig(): return "".join(rng.choice("0123456789ABCDEF") for _ in range(10))
def ctr(): return f"CTR/{rng.randint(2022,2026)}/{rng.randint(1,9999):05d}"
COMP = ["Nordtech S.r.l.", "Valdarno Servizi S.p.A.", "Edilgenova S.n.c.", "Alpi Energia S.r.l.", "Meridiana Logistica S.p.A.", "Studio Bianchi & Associati", "Cantieri del Tirreno S.r.l."]
COMP_EN = ["Northbridge Ltd.", "Harborline Services Inc.", "Granite & Co. LLP", "Meridian Logistics Ltd.", "Oakfield Energy plc", "Thames Works Ltd."]
class Ent(list):
    def add(self, kind, text): self.append({"type": kind, "text": text}); return text

SENT = {"it": ["La Società si impegna a fornire i servizi entro trenta giorni dalla sottoscrizione, salvo cause di forza maggiore documentate.",
  "Il corrispettivo è dovuto a mezzo bonifico bancario entro sessanta giorni dalla data di ricevimento della fattura.",
  "In caso di ritardo nel pagamento decorreranno gli interessi di mora nella misura prevista dal D.Lgs. 231/2002.",
  "Le parti dichiarano di aver letto e approvato specificamente le clausole di cui agli articoli 7, 9 e 12.",
  "Per ogni controversia sarà competente in via esclusiva il Foro di Genova, con rinuncia a qualsiasi altro foro.",
  "Il Fornitore garantisce la conformità dei beni alle specifiche tecniche allegate e alla normativa vigente."],
 "en": ["The Supplier shall deliver the services within thirty days of signature, save for documented force majeure events.",
  "The fee is payable by bank transfer within sixty days from the date the invoice is received.",
  "In the event of late payment, default interest shall accrue at the statutory rate in force at the due date.",
  "The parties declare that they have read and specifically approved the clauses set out in articles 7, 9 and 12.",
  "Any dispute shall be subject to the exclusive jurisdiction of the courts of London, excluding any other forum.",
  "The Supplier warrants that the goods conform to the attached technical specifications and applicable law."]}
TITLE = {"it": ["Contratto di fornitura", "Verbale di consegna", "Convenzione quadro", "Ordine di acquisto", "Fattura commerciale"],
         "en": ["Supply Agreement", "Delivery Report", "Framework Agreement", "Purchase Order", "Commercial Invoice"]}

def comp(lang): return rng.choice(COMP if lang == "it" else COMP_EN)

def wrap(d, text, f, width):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: out.append(cur); cur = w
    return out + [cur]

# ---------------------------------------------------------------- templates: return (img, gt_lines, entities)
def base(): img = Image.new("RGB", (1240, 1754), "white"); return img, ImageDraw.Draw(img)

def sentence_with_entity(lang, ent):
    k = rng.randint(0, 5)
    if k == 0: return (f"Il corrispettivo complessivo è pari a {ent.add('amount', amt(lang, rng.uniform(1e3, 9e6)))}, oltre IVA." if lang == "it" else f"The total consideration is {ent.add('amount', amt(lang, rng.uniform(1e3, 9e6)))}, plus VAT.")
    if k == 1: return (f"Il pagamento sarà effettuato sul conto IBAN {ent.add('iban', iban())} intestato a {comp(lang)}." if lang == "it" else f"Payment shall be made to account IBAN {ent.add('iban', iban())} held by {comp(lang)}.")
    if k == 2: return (f"Il presente contratto, n. {ent.add('contract', ctr())}, è stipulato in data {ent.add('date', date(lang))}." if lang == "it" else f"This agreement, no. {ent.add('contract', ctr())}, is entered into on {ent.add('date', date(lang))}.")
    if k == 3: return (f"{comp(lang)}, P.IVA {ent.add('vat', piva())}, CIG {ent.add('cig', cig())}, è l'appaltatore." if lang == "it" else f"{comp(lang)}, VAT no. {ent.add('vat', piva())}, ref. {ent.add('cig', cig())}, is the contractor.")
    if k == 4: return (f"Il legale rappresentante, C.F. {ent.add('cf', cf())}, firma per accettazione in data {ent.add('date', date(lang))}." if lang == "it" else f"The legal representative, tax code {ent.add('cf', cf())}, signs for acceptance on {ent.add('date', date(lang))}.")
    return (f"La penale giornaliera è fissata in {ent.add('amount', amt(lang, rng.uniform(50, 5000)))} per ogni giorno di ritardo." if lang == "it" else f"The daily penalty is set at {ent.add('amount', amt(lang, rng.uniform(50, 5000)))} for each day of delay.")

def t_prose(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y), title, font=F(fn, 44*s, True), fill="black"); gt.append(title); y += int(100*s)
    f = F(fn, 30*s); lh = int(30*s*1.45)
    for _ in range(5):
        p = " ".join([sentence_with_entity(lang, ent), rng.choice(SENT[lang]), sentence_with_entity(lang, ent)])
        for ln in wrap(d, p, f, 1040): d.text((100, y), ln, font=f, fill="black"); y += lh
        gt.append(p); y += lh // 2
    return img, gt, ent

def t_twocol(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y0 = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y0), title, font=F(fn, 40*s, True), fill="black"); gt.append(title); y0 += int(90*s)
    f = F(fn, 26*s); lh = int(26*s*1.4); cw = 490; ps = [" ".join([sentence_with_entity(lang, ent), rng.choice(SENT[lang])]) for _ in range(8)]
    for c in range(2):
        y = y0
        for p in ps[4*c:4*c+4]:
            for ln in wrap(d, p, f, cw): d.text((100 + c*(cw+60), y), ln, font=f, fill="black"); y += lh
            y += lh // 2
    return img, gt + ps, ent

def t_table(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y), title, font=F(fn, 40*s, True), fill="black"); gt.append(title); y += int(90*s)
    hdr = ["Voce", "Q.tà", "Prezzo unit.", "Importo"] if lang == "it" else ["Item", "Qty", "Unit price", "Amount"]
    cols = [100, 600, 720, 920, 1140]; rh = int(54*s); f = F(fn, 26*s); fb = F(fn, 26*s, True)
    rows = [hdr]; tot = 0
    for i in range(var(12, 8, 16)):
        q = rng.randint(1, 40); p = round(rng.uniform(3, 900), 2); tot += q*p
        rows.append([f"{rng.choice(['Cavo','Valvola','Pannello','Sensore','Staffa','Guarnizione','Modulo','Inverter'] if lang=='it' else ['Cable','Valve','Panel','Sensor','Bracket','Gasket','Module','Inverter'])} {rng.choice('ABCDEFGH')}{rng.randint(100,999)}", str(q), amt(lang, p), amt(lang, q*p)])
    rows.append([("Totale" if lang == "it" else "Total"), "", "", ent.add("amount", amt(lang, tot))])
    for r, row in enumerate(rows):
        d.rectangle([100, y, 1140, y+rh], outline="black", width=2)
        for c, cell in enumerate(row):
            if c: d.line([cols[c], y, cols[c], y+rh], fill="black", width=2)
            ff = fb if r in (0, len(rows)-1) else f; w = d.textlength(cell, font=ff)
            x = cols[c] + 10 if c == 0 else cols[c+1] - 10 - w
            d.text((x, y + rh*0.22), cell, font=ff, fill="black")
            if cell: gt.append(cell)
            if c == 3 and 0 < r < len(rows)-1: ent.add("amount", cell)
        y += rh
    return img, gt, ent

def t_form(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y), title, font=F(fn, 44*s, True), fill="black"); gt.append(title); y += int(110*s)
    L = (["Ragione sociale", "Partita IVA", "Codice fiscale", "IBAN", "Numero contratto", "CIG", "Data stipula", "Importo", "Scadenza", "Referente"] if lang == "it"
         else ["Company name", "VAT number", "Tax code", "IBAN", "Contract number", "Reference", "Signing date", "Amount", "Expiry date", "Contact person"])
    V = [comp(lang), ent.add("vat", piva()), ent.add("cf", cf()), ent.add("iban", iban()), ent.add("contract", ctr()), ent.add("cig", cig()),
         ent.add("date", date(lang)), ent.add("amount", amt(lang, rng.uniform(1e3, 5e6))), ent.add("date", date(lang)), rng.choice(["Marco Rossi", "Giulia Verdi", "Paolo Neri", "Anna Conti"])]
    fl, fv = F(fn, 28*s), F(fn, 30*s, True); rh = int(95*s)
    for l, v in zip(L, V):
        d.text((100, y), l.upper(), font=F(fn, 20*s), fill=(60, 60, 60)); d.text((100, y + int(30*s)), v, font=fv, fill="black")
        d.line([100, y + rh - 12, 1140, y + rh - 12], fill=(120, 120, 120), width=1); gt += [l.upper(), v]; y += rh
    return img, gt, ent

def t_invoice(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(100*s)
    f, fb, fs = F(fn, 26*s), F(fn, 26*s, True), F(fn, 22*s)
    a, b = comp(lang), comp(lang)
    blocks = [(100, [a, f"P.IVA {ent.add('vat', piva())}", "Via Garibaldi 12, 16124 Genova"]), (700, [b, f"P.IVA {ent.add('vat', piva())}", "Corso Italia 45, 20122 Milano"])]
    for x, ls in blocks:
        yy = y
        for i, ln in enumerate(ls): d.text((x, yy), ln, font=fb if i == 0 else f, fill="black"); yy += int(36*s); gt.append(ln)
    y += int(170*s); head = f"{'Fattura' if lang=='it' else 'Invoice'} n. {ent.add('contract', ctr())} - {ent.add('date', date(lang))}"
    d.text((100, y), head, font=F(fn, 36*s, True), fill="black"); gt.append(head); y += int(90*s)
    rh = int(46*s); tot = 0
    for i in range(var(9, 6, 14)):
        q = rng.randint(1, 20); p = round(rng.uniform(5, 700), 2); tot += q*p
        desc = f"{'Servizio' if lang=='it' else 'Service'} {rng.choice('ABCDEFGH')}{rng.randint(10,99)}/{rng.randint(2022,2026)}"
        for x, t in [(100, desc), (640, str(q)), (760, amt(lang, p)), (960, amt(lang, q*p))]:
            d.text((x, y), t, font=f, fill="black"); gt.append(t)
        ent.add("amount", amt(lang, q*p)); y += rh
    d.line([100, y, 1140, y], fill="black", width=2); y += 14
    iva = tot*0.22; lines = [("Imponibile" if lang == "it" else "Subtotal", tot), ("IVA 22%" if lang == "it" else "VAT 22%", iva), ("TOTALE" if lang == "it" else "TOTAL", tot + iva)]
    for l, v in lines:
        t = ent.add("amount", amt(lang, v)); d.text((700, y), l, font=fb, fill="black"); d.text((960, y), t, font=fb, fill="black"); gt += [l, t]; y += int(44*s)
    y += int(60*s); foot = f"IBAN {ent.add('iban', iban())}"; d.text((100, y), foot, font=f, fill="black"); gt.append(foot); y += int(50*s)
    note = rng.choice(SENT[lang]); [d.text((100, y + i*int(30*s)), ln, font=fs, fill="black") for i, ln in enumerate(wrap(d, note, fs, 1040))]; gt.append(note)
    return img, gt, ent

def t_footnotes(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y), title, font=F(fn, 42*s, True), fill="black"); gt.append(title); y += int(100*s)
    f = F(fn, 28*s); lh = int(28*s*1.45)
    for i in range(3):
        p = f"{sentence_with_entity(lang, ent)} {rng.choice(SENT[lang])}"
        for ln in wrap(d, p + f" [{i+1}]", f, 1040): d.text((100, y), ln, font=f, fill="black"); y += lh
        gt.append(p + f" [{i+1}]"); y += lh // 2
    fpt = var(17, 14, 22); y = 1330; d.line([100, y, 500, y], fill="black", width=2); y += 14; fsm = F(fn, fpt*s); lh = int(fpt*s*1.4)
    for i in range(var(5, 3, 6)):
        p = f"[{i+1}] {sentence_with_entity(lang, ent)}"
        for ln in wrap(d, p, fsm, 1040): d.text((100, y), ln, font=fsm, fill="black"); y += lh
        gt.append(p)
    return img, gt, ent

def t_bilingual(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y0 = int(110*s)
    d.text((100, y0), "Contratto / Agreement", font=F(fn, 40*s, True), fill="black"); gt.append("Contratto / Agreement"); y0 += int(90*s)
    f = F(fn, 24*s); lh = int(24*s*1.4); cw = 490; left, right = [], []
    for i in range(5):
        v = rng.randint(1000, 900000) + rng.randint(0, 99)/100; dt = (rng.randint(1, 28), rng.randint(1, 12), rng.randint(2022, 2026)); n = rng.randint(1, 9999); k = rng.randint(0, 2)
        if k == 0:
            e1, e2 = ent.add("amount", amt("it", v)), ent.add("amount", amt("en", v)); left.append(f"Il corrispettivo è di {e1}."); right.append(f"The fee is {e2}.")
        elif k == 1:
            e1, e2 = ent.add("date", f"{dt[0]:02d}/{dt[1]:02d}/{dt[2]}"), ent.add("date", f"{dt[2]}-{dt[1]:02d}-{dt[0]:02d}"); left.append(f"Data di consegna: {e1}."); right.append(f"Delivery date: {e2}.")
        else:
            e = ent.add("contract", f"CTR/2025/{n:05d}"); left.append(f"Riferimento {e}, soggetto a Foro di Genova."); right.append(f"Reference {e}, subject to the courts of London.")
        left.append(rng.choice(SENT["it"])); right.append(rng.choice(SENT["en"]))
    for c, ps in enumerate([left, right]):
        y = y0
        for p in ps:
            for ln in wrap(d, p, f, cw): d.text((100 + c*(cw+60), y), ln, font=f, fill="black"); y += lh
            y += lh // 2
    return img, gt + left + right, ent

def t_clauses(lang, s, fn):
    img, d = base(); ent = Ent(); gt = []; y = int(110*s); title = rng.choice(TITLE[lang])
    d.text((100, y), title, font=F(fn, 42*s, True), fill="black"); gt.append(title); y += int(95*s)
    fh, f = F(fn, 28*s, True), F(fn, 26*s); lh = int(26*s*1.4)
    for i in range(1, 6):
        h = f"{'Art.' if lang=='it' else 'Article'} {i} - {rng.choice(['Oggetto','Durata','Corrispettivo','Recesso','Foro competente'] if lang=='it' else ['Subject matter','Term','Consideration','Termination','Jurisdiction'])}"
        d.text((100, y), h, font=fh, fill="black"); gt.append(h); y += int(40*s)
        for j in range(2):
            p = f"{i}.{j+1} " + (sentence_with_entity(lang, ent) if j == 0 else rng.choice(SENT[lang]))
            for ln in wrap(d, p, f, 1000): d.text((140, y), ln, font=f, fill="black"); y += lh
            gt.append(p); y += lh // 3
        y += int(20*s)
    return img, gt, ent

TEMPLATES = [("prose", t_prose), ("twocol", t_twocol), ("table", t_table), ("form", t_form), ("invoice", t_invoice),
             ("footnotes", t_footnotes), ("bilingual", t_bilingual), ("clauses", t_clauses)]
SCALE = {1: 1.0, 2: 0.92, 3: 0.82, 4: 0.72, 5: 0.62}

# ---------------------------------------------------------------- degradations
def perspective(img, mag):
    w, h = img.size; j = lambda: rng.uniform(-mag, mag)
    src = [(0 + j()*w, 0 + j()*h), (w + j()*w, 0 + j()*h), (w + j()*w, h + j()*h), (0 + j()*w, h + j()*h)]
    A = []
    for (x, y), (u, v) in zip([(0, 0), (w, 0), (w, h), (0, h)], src):
        A += [[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]]
    coef = np.linalg.solve(np.array(A, float), np.array([c for p in src for c in p], float))
    return img.transform((w, h), Image.PERSPECTIVE, tuple(coef), Image.BICUBIC, fillcolor=(235, 235, 235))

def stamp(img):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov); cx, cy, r = rng.randint(300, 900), rng.randint(500, 1300), 150
    d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(170, 30, 40, 150), width=8); d.ellipse([cx-r+18, cy-r+18, cx+r-18, cy+r-18], outline=(170, 30, 40, 150), width=3)
    f = ImageFont.truetype(SS + "Courier New Bold.ttf", 30); d.text((cx-85, cy-35), "PROTOCOLLO", font=f, fill=(170, 30, 40, 170)); d.text((cx-60, cy+5), "N. 4471", font=f, fill=(170, 30, 40, 170))
    ov = ov.rotate(rng.uniform(-25, 25), center=(cx, cy), resample=Image.BICUBIC)
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

def hand_note(img, ent, lang):
    t = amt(lang, rng.uniform(100, 9000)); d = ImageDraw.Draw(img); f = ImageFont.truetype(HAND, 46)
    x, y = rng.randint(650, 800), rng.randint(1560, 1640); d.text((x, y), t, font=f, fill=(20, 30, 120)); ent.add("hand", t)

def degrade(img, level, idx):
    if level == 1: return img
    p = {2: dict(rot=0.7, blur=0.5, noise=4, con=0.95, q=75, down=1.0, warp=0.0),
         3: dict(rot=1.8, blur=0.8, noise=10, con=0.8, q=50, down=0.75, warp=0.004),
         4: dict(rot=2.5, blur=1.0, noise=18, con=0.65, q=35, down=0.6, warp=0.008),
         5: dict(rot=3.0, blur=1.2, noise=26, con=0.5, q=25, down=0.5, warp=0.014)}[level]
    if level >= 4: img = stamp(img)
    if p["warp"]: img = perspective(img, p["warp"])
    img = img.rotate(rng.uniform(-p["rot"], p["rot"]), fillcolor=(236, 236, 236), resample=Image.BICUBIC, expand=False)
    img = ImageEnhance.Contrast(img).enhance(p["con"]).filter(ImageFilter.GaussianBlur(p["blur"]))
    a = np.asarray(img, dtype=np.float32)
    yy = np.linspace(0, 1, a.shape[0])[:, None, None]; a = a * (0.93 + 0.07 * (1 - yy)) if level >= 3 else a  # lighting gradient
    if level == 5:  # fold shadow + vignette
        xx = np.linspace(-1, 1, a.shape[1])[None, :, None]; a = a * (1 - 0.18 * np.exp(-((xx - 0.1) / 0.03) ** 2)); a = a * (1 - 0.25 * np.clip(np.abs(xx) - 0.7, 0, 1))
    a = a + np.random.default_rng(idx).normal(0, p["noise"], a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if p["down"] < 1: w, h = img.size; img = img.resize((int(w*p["down"]), int(h*p["down"])), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    import io
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=p["q"]); img = Image.open(buf).convert("RGB")
    if level == 5 and idx % 2 == 1: img = img.rotate(90, expand=True, fillcolor=(236, 236, 236))
    return img

meta = []; n = 0; W = 2 if ARGS.reps * 40 <= 100 else 3
for level, rep_ in ((l, r) for l in range(1, 6) for r in range(ARGS.reps)):
    for t, (name, fn_) in enumerate(TEMPLATES):
        lang = "it" if (t + level + rep_) % 2 == 0 else "en"; font = rng.choice(list(FONTS))
        img, gt, ent = fn_(lang, SCALE[level], font)
        if level >= 3 and name in ("prose", "form", "invoice", "clauses"): hand_note(img, ent, lang)
        img = degrade(img, level, n)
        id_ = f"{n:0{W}d}_{name}_L{level}_{lang}"; img.save(OUT / f"{id_}.png")
        (OUT / f"{id_}.txt").write_text("\n".join(gt), encoding="utf-8")
        meta.append({"id": id_, "gt": "exact", "level": level, "tpl": name, "lang": lang, "font": font, "rot90": bool(level == 5 and n % 2 == 1),
                     "entities": list(ent)}); n += 1
json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
print(n, "pages;", sum(len(m["entities"]) for m in meta), "entities")
