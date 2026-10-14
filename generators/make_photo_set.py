"""python generators/make_photo_set.py : twelve Italian forms to print and photograph -> photo_set/
  print.pdf          the twelve pages, A4 at 300 dpi, to print as they are (no scaling)
  <ID>.png, <ID>.txt the digital page and its exact text; meta.json with the entities (every value written in the form)
T01..T10 are for training, X01 and X02 for testing only: their photos never enter a training set.
What the pages are for: comb fields (one character per box: tax codes, IBANs, VAT numbers, dates, postcodes, phone numbers),
where release v1.0.0 loops on real forms (docs/LEDGER.md, XFUND), and a wide Italian vocabulary: the twelve documents are of
twelve different kinds, and check_vocabulary() fails if a content word of six letters or more appears in two of them, apart
from the field names every form shares (SHARED).
Ground truth convention, the one the model is trained to write: one line per printed line, top to bottom; a field is its label,
a space and its value, with the characters of a comb field written together ("Codice fiscale RSSMRA85T10H501Z"); a tick box is
"☒ text" when ticked and "☐ text" when not. Values are made up; tax codes, IBANs and VAT numbers carry valid check characters.
The code in the top right corner (T01 ...) is printed so that a photo can be matched to its page; it is part of the text."""
import json, os, pathlib, random, re, string
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path("photo_set"); OUT.mkdir(exist_ok=True)
W, H, M = 2480, 3508, 170   # A4 at 300 dpi, margin
S = "/System/Library/Fonts/"; SS = S + "Supplemental/"
FACES = {"helv": (S + "Helvetica.ttc", 0, 1), "times": (S + "Times.ttc", 0, 1), "georgia": (SS + "Georgia.ttf", 0, None),
         "verdana": (SS + "Verdana.ttf", 0, None), "trebuchet": (SS + "Trebuchet MS.ttf", 0, None)}
BOLDF = {"georgia": SS + "Georgia Bold.ttf", "verdana": SS + "Verdana Bold.ttf", "trebuchet": SS + "Trebuchet MS Bold.ttf"}
PEN = [SS + "Bradley Hand Bold.ttf", SS + "Courier New Bold.ttf", S + "Helvetica.ttc"]


def font(face, size, bold=False):
    p, i, b = FACES[face]
    if bold:
        return ImageFont.truetype(p, size, index=b) if b is not None else ImageFont.truetype(BOLDF[face], size)
    return ImageFont.truetype(p, size, index=i)


# ---------------------------------------------------------------- valid identifiers
ODD = dict(zip(string.digits + string.ascii_uppercase, [1, 0, 5, 7, 9, 13, 15, 17, 19, 21] + [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]))
EVEN = {c: (int(c) if c.isdigit() else ord(c) - 65) for c in string.digits + string.ascii_uppercase}


def codice_fiscale(rng, surname, name, year, month, day, female, place):
    cons = lambda s: [c for c in s.upper() if c.isalpha() and c not in "AEIOU"]
    vow = lambda s: [c for c in s.upper() if c in "AEIOU"]
    sur = (cons(surname) + vow(surname) + ["X"] * 3)[:3]
    c = cons(name)
    nam = [c[0], c[2], c[3]] if len(c) >= 4 else (c + vow(name) + ["X"] * 3)[:3]
    body = "".join(sur + nam) + f"{year % 100:02d}" + "ABCDEHLMPRST"[month - 1] + f"{day + 40 * female:02d}" + place
    s = sum(ODD[ch] if k % 2 == 0 else EVEN[ch] for k, ch in enumerate(body))
    return body + chr(65 + s % 26)


def iban_it(rng):
    abi, cab = f"{rng.randrange(1000, 99999):05d}", f"{rng.randrange(1000, 99999):05d}"
    conto = "".join(rng.choice(string.digits) for _ in range(12))
    vals = {**{d: int(d) for d in string.digits}, **{ch: v for ch, v in zip(string.ascii_uppercase, [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23])}}
    # CIN: Italian national check letter over ABI + CAB + account
    odd = [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]
    bban = abi + cab + conto
    s = sum((odd[int(ch)] if k % 2 == 0 else int(ch)) for k, ch in enumerate(bban))
    cin = chr(65 + s % 26)
    bban = cin + bban
    num = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in bban + "IT00")
    check = 98 - int(num) % 97
    return f"IT{check:02d}{bban}"


def partita_iva(rng):
    d = [rng.randrange(10) for _ in range(10)]
    s = sum(x if k % 2 == 0 else (2 * x if 2 * x < 10 else 2 * x - 9) for k, x in enumerate(d))
    return "".join(map(str, d)) + str((10 - s % 10) % 10)


# ---------------------------------------------------------------- page drawing
class Page:
    def __init__(self, code, face, rng, k=1.0, style=None):
        self.style = style or {}   # layout of this document: title, headings, field pairs, block order (STYLES)
        self.pending, self.b_done = None, False
        self.k = k   # every size on the page is multiplied by k, so that the text fills the page
        self.img = Image.new("RGB", (W, H), "white"); self.d = ImageDraw.Draw(self.img)
        self.face, self.rng, self.y, self.lines, self.entities = face, rng, M, [], []
        self.pen = rng.choice(PEN)
        f = font("helv", 64, bold=True)
        self.d.rectangle([W - M - 250, M - 40, W - M, M + 60], outline="black", width=5)
        self.d.text((W - M - 225, M - 30), code, font=f, fill="black")
        self.lines.append(code); self.y = M + 120

    def title(self, text, size=62):
        self.flush()
        size = int(size * self.k)
        f = font(self.face, size, bold=True)
        while self.d.textlength(text, font=f) > W - 2 * M - 300:   # clear of the code box, inside the margins
            size -= 2; f = font(self.face, size, bold=True)
        w = self.d.textlength(text, font=f)
        if self.style.get("title") == "left":
            self.d.text((M, self.y), text, font=f, fill="black"); self.y += size + 14
            self.d.line([M, self.y, W - M - 300 if self.y < M + 300 else W - M, self.y], fill="black", width=4); self.y += 16
        else:
            self.d.text(((W - w) / 2, self.y), text, font=f, fill="black"); self.y += size + 26
        self.lines.append(text)

    def para(self, text, size=40, gap=14):
        self.flush()
        size = int(size * self.k)
        f = font(self.face, size); words = text.split(); line = ""
        for wd in words:
            t = (line + " " + wd).strip()
            if self.d.textlength(t, font=f) > W - 2 * M:
                self.d.text((M, self.y), line, font=f, fill="black"); self.lines.append(line); self.y += size + gap; line = wd
            else:
                line = t
        if line:
            self.d.text((M, self.y), line, font=f, fill="black"); self.lines.append(line); self.y += size + gap
        self.y += 10

    def flush(self):
        if self.pending:
            item, self.pending = self.pending, None
            self.comb(*item, alone=True)

    def comb(self, label, value, kind, size=40, box=62, x=None, newline=True, alone=False):
        """label then one box per character of value, the value written in the boxes by 'hand'"""
        if newline and not alone and self.style.get("pair") and len(value) <= 11:   # short fields two to a line in this layout
            if self.pending is None:
                self.pending = (label, value, kind); return None
            first, self.pending = self.pending, None
            return self.comb_row([first, (label, value, kind)])
        if newline:
            self.flush()
        size, box = int(size * self.k), int(box * self.k)
        f = font(self.face, size); x0 = M if x is None else x
        self.d.text((x0, self.y + 12), label, font=f, fill="black")
        bx = x0 + self.d.textlength(label, font=f) + 24
        units = sum(1 if ch not in " /" else 0.5 for ch in value) + 0.1 * value.count("/")
        if newline and (W - M - bx) / units < 0.8 * box:   # a long field goes under its label, at full width
            self.y += size + 18; bx = M
        box = int(min(box, (W - M - bx) / units))   # narrower boxes rather than boxes off the page
        assert box >= 34, f"comb field '{label}' does not fit on the line"
        pf = ImageFont.truetype(self.pen, int(box * 0.72), index=0)
        for k, ch in enumerate(value):
            if ch == " ":
                bx += box // 2; continue
            if ch == "/":   # printed between groups of boxes, as on real forms
                self.d.text((bx + 6, self.y + 8), "/", font=font(self.face, int(box * 0.8)), fill="black"); bx += box // 2 + 6; continue
            self.d.rectangle([bx, self.y, bx + box, self.y + box], outline="black", width=3)
            w = self.d.textlength(ch, font=pf)
            self.d.text((bx + (box - w) / 2 + self.rng.uniform(-3, 3), self.y + 4 + self.rng.uniform(-3, 3)), ch, font=pf, fill=(20, 30, 110))
            bx += box
        assert bx <= W - M + 1, f"comb field '{label}' runs past the margin"
        compact = value.replace(" ", "")
        self.entities.append({"type": kind, "text": compact})
        if newline:
            self.lines.append(f"{label} {compact}"); self.y += box + 30
        return bx, f"{label} {compact}"

    def comb_row(self, items, size=40, box=58):
        """two or three comb fields on one printed line; a field that would not fit at full size starts a new line"""
        self.flush()
        x, parts = M, []
        for label, value, kind in items:
            f = font(self.face, int(size * self.k)); bk = int(box * self.k)
            units = sum(1 if ch not in " /" else 0.5 for ch in value) + 0.1 * value.count("/")
            if parts and x + self.d.textlength(label, font=f) + 24 + units * bk > W - M:
                self.lines.append(" ".join(parts)); self.y += bk + 30; x, parts = M, []
            bx, txt = self.comb(label, value, kind, size, box, x=x, newline=False)
            parts.append(txt); x = bx + 60
        self.lines.append(" ".join(parts)); self.y += int(box * self.k) + 30

    def line_field(self, label, value, kind, size=40):
        self.flush()
        size = int(size * self.k)
        f = font(self.face, size); self.d.text((M, self.y), label, font=f, fill="black")
        x = M + self.d.textlength(label, font=f) + 20
        self.d.line([x, self.y + size + 6, W - M, self.y + size + 6], fill="black", width=2)
        pf = ImageFont.truetype(self.pen, int(size * 1.05), index=0)
        self.d.text((x + 10, self.y - 8), value, font=pf, fill=(20, 30, 110))
        self.entities.append({"type": kind, "text": value}); self.lines.append(f"{label} {value}"); self.y += size + 34

    def heading(self, text, size=46):
        self.flush()
        size = int(size * self.k)
        self.y += 16; f = font(self.face, size, bold=True); kind = self.style.get("head", "plain")
        if kind == "band":
            self.d.rectangle([M, self.y - 8, W - M, self.y + size + 14], fill=(215, 215, 215))
        elif kind == "box":
            self.d.rectangle([M, self.y - 8, W - M, self.y + size + 14], outline="black", width=3)
        self.d.text((M + (16 if kind != "plain" else 0), self.y), text, font=f, fill="black"); self.lines.append(text); self.y += size + 22 + (14 if kind != "plain" else 0)

    def table(self, header, rows, size=36):
        self.flush()
        size = int(size * self.k)
        """a ruled table; written in the text as the model writes tables: cells joined by " | ", a rule line after the header"""
        f, fb = font(self.face, size), font(self.face, size, bold=True)
        widths = [max(self.d.textlength(str(r[c]), font=fb) for r in [header] + rows) + 40 for c in range(len(header))]
        scale = (W - 2 * M) / sum(widths); widths = [w * scale for w in widths]
        rh = size + 26
        for k, row in enumerate([header] + rows):
            x = M
            for c, cell in enumerate(row):
                self.d.rectangle([x, self.y, x + widths[c], self.y + rh], outline="black", width=2)
                self.d.text((x + 14, self.y + 12), str(cell), font=fb if k == 0 else f, fill="black"); x += widths[c]
            self.lines.append(" | ".join(map(str, row)))
            if k == 0:
                self.lines.append(" | ".join("---" for _ in row))
            self.y += rh
        self.y += 30
        for row in rows:
            for cell in row:
                if re.search(r"\d", str(cell)): self.entities.append({"type": "table", "text": str(cell)})

    def ticks(self, options, ticked, size=40):
        self.flush()
        if self.style.get("table_first") and not self.b_done:
            section_b(self)
        size = int(size * self.k)
        f = font(self.face, size)
        for k, opt in enumerate(options):
            self.d.rectangle([M, self.y + 4, M + size, self.y + 4 + size], outline="black", width=3)
            if k in ticked:
                self.d.line([M + 6, self.y + 10, M + size - 6, self.y + size - 2], fill=(20, 30, 110), width=6)
                self.d.line([M + size - 6, self.y + 10, M + 6, self.y + size - 2], fill=(20, 30, 110), width=6)
            self.d.text((M + size + 24, self.y), opt, font=f, fill="black")
            self.lines.append(("☒ " if k in ticked else "☐ ") + opt); self.y += size + 22
        self.y += 8

    def sign(self, place_date, size=40):
        self.flush()
        size = int(size * self.k)
        f = font(self.face, size); self.y += 20
        self.d.text((M, self.y), place_date, font=f, fill="black")
        self.d.text((W - M - 700, self.y), "Firma", font=f, fill="black")
        self.d.line([W - M - 580, self.y + size, W - M, self.y + size], fill="black", width=2)
        self.lines.append(f"{place_date} Firma"); self.y += size + 30

    def check(self, code):
        assert self.y < H - M, f"{code} runs off the page ({self.y} px)"


# ---------------------------------------------------------------- the twelve documents
P = lambda rng: f"3{rng.randrange(20, 99)}{rng.randrange(1000000, 9999999)}"
D = lambda d, m, y: f"{d:02d}/{m:02d}/{y}"


def doc(code, face, seed, build):
    """drawn once to measure, then again with every size scaled so that the page is about 85% full (at most 1.6 times)"""
    style = STYLES[code]
    rng = random.Random(seed); p = Page(code, face, rng, style=style); build(p, rng); p.flush()
    k = min(1.6, (0.85 * H - M) / (p.y - M))
    rng = random.Random(seed); p = Page(code, face, rng, k, style); build(p, rng); p.flush(); p.check(code)
    return p


def t01(p, r):
    p.title("COMUNE DI CASTELNUOVO DEL GARDA"); p.title("Dichiarazione sostitutiva di certificazione anagrafica", 46)
    p.para("Il sottoscritto, consapevole delle sanzioni penali previste per chi rende dichiarazioni mendaci, dichiara sotto la propria responsabilità di essersi trasferito presso il nuovo alloggio indicato qui sotto, unitamente ai familiari conviventi elencati nel riquadro successivo.")
    p.line_field("Cognome e nome", "Bertolini Gianmaria", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Bertolini", "Gianmaria", 1979, 3, 14, 0, "C225"), "cf")
    p.comb_row([("Nato il", D(14, 3, 1979), "date"), ("CAP", "37014", "cap")])
    p.line_field("Indirizzo", "Via delle Ortensie 27, interno 4", "address")
    p.comb("Telefono", P(r), "phone")
    p.ticks(["Proveniente da altro comune italiano", "Proveniente dall'estero", "Cambio di abitazione nello stesso comune"], {0})
    p.para("Familiari conviventi: la coniuge Lucrezia, nata nel 1983, e il figlio Tommaso, iscritto alla scuola primaria cittadina. Si allega fotocopia del documento d'identità in corso di validità.")
    section_b(p); p.sign("Castelnuovo del Garda, 3 febbraio 2026")


def t02(p, r):
    p.title("BANCA POPOLARE DELLA VALDELSA"); p.title("Disposizione di bonifico SEPA", 50)
    p.para("Ordinante titolare del rapporto, con addebito sul conto corrente indicato. La valuta beneficiario sarà riconosciuta entro la giornata operativa successiva all'esecuzione dell'ordine.")
    p.line_field("Ordinante", "Ferraguti Ornella", "name")
    p.comb("IBAN ordinante", iban_it(r), "iban")
    p.line_field("Beneficiario", "Officine Meccaniche Zanardi S.r.l.", "company")
    p.comb("IBAN beneficiario", iban_it(r), "iban")
    p.comb("Partita IVA beneficiario", partita_iva(r), "vat")
    p.comb_row([("Importo euro", "1.846,50", "amount"), ("Data esecuzione", D(17, 11, 2025), "date")])
    p.line_field("Causale", "Saldo fattura 412/B riparazione tornio", "text")
    p.ticks(["Bonifico ordinario", "Bonifico istantaneo", "Giroconto tra rapporti"], {1})
    p.para("Commissione applicata secondo il prospetto informativo vigente. L'ordine diventa irrevocabile dopo la conferma mediante codice dispositivo.")
    section_b(p); p.sign("Poggibonsi, 16 novembre 2025")


def t03(p, r):
    p.title("UNIVERSITÀ DEGLI STUDI DI MACERATA"); p.title("Domanda di immatricolazione", 50)
    p.para("Corso di laurea magistrale in Filologia classica e moderna. La studentessa chiede l'ammissione al primo anno accademico e dichiara di possedere il titolo triennale conseguito con votazione finale espressa in centodecimi.")
    p.line_field("Cognome e nome", "Quaranta Benedetta", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Quaranta", "Benedetta", 2001, 9, 22, 1, "E783"), "cf")
    p.comb_row([("Matricola", "M8812047", "id"), ("Voto di laurea", "108", "number")])
    p.comb("Data di nascita", D(22, 9, 2001), "date")
    p.comb("Cellulare", P(r), "phone")
    p.line_field("Posta elettronica", "b.quaranta@studenti.unimc.example", "email")
    p.ticks(["Tempo pieno", "Tempo parziale", "Richiesta di riconoscimento esami pregressi"], {0, 2})
    p.para("Allegati: diploma supplement, attestato di versamento ERSU, fototessera digitale. Eventuali borse di merito verranno erogate dopo la verifica dei requisiti reddituali.")
    section_b(p); p.sign("Macerata, 28 agosto 2026")


def t04(p, r):
    p.title("MUTUA SANITARIA INTEGRATIVA"); p.title("Richiesta di rimborso prestazioni odontoiatriche", 46)
    p.para("L'assistito domanda il rimborso delle spese sostenute presso lo studio dentistico convenzionato per igiene professionale, otturazione di due molari e radiografia panoramica dell'arcata superiore.")
    p.line_field("Assistito", "Scognamiglio Pasquale", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Scognamiglio", "Pasquale", 1966, 12, 5, 0, "F839"), "cf")
    p.comb("Numero tessera", "40027781936", "id")
    p.comb_row([("Totale fatture", "742,00", "amount"), ("Data prestazione", D(9, 6, 2026), "date")])
    p.comb("IBAN per l'accredito", iban_it(r), "iban")
    p.ticks(["Prestazione in forma diretta", "Prestazione in forma indiretta", "Ricovero ospedaliero"], {1})
    p.para("Documentazione da trasmettere: fatture quietanzate in originale, prescrizione del medico curante, piano di cura firmato dall'odontoiatra. Franchigia annuale pari a centocinquanta euro.")
    section_b(p); p.sign("Napoli, 14 giugno 2026")


def t05(p, r):
    p.title("POSTE E SPEDIZIONI"); p.title("Delega al ritiro di corrispondenza e pacchi", 48)
    p.para("Con la presente autorizzo la persona sotto indicata a ritirare in mia vece le raccomandate giacenti presso l'ufficio postale di quartiere, sollevando lo sportellista da qualunque contestazione futura.")
    p.line_field("Delegante", "Vitaliano Ermenegildo", "name")
    p.comb("Codice fiscale delegante", codice_fiscale(r, "Vitaliano", "Ermenegildo", 1948, 1, 30, 0, "L219"), "cf")
    p.line_field("Delegato", "Vitaliano Rosalba", "name")
    p.comb("Codice fiscale delegato", codice_fiscale(r, "Vitaliano", "Rosalba", 1975, 7, 8, 1, "L219"), "cf")
    p.comb_row([("Avviso di giacenza", "AG55190274", "id"), ("Scadenza", D(4, 12, 2025), "date")])
    p.comb("Documento delegato", "CA48261KX", "id")
    p.ticks(["Raccomandata", "Assicurata", "Pacco standard"], {0, 2})
    p.para("Occorre esibire l'originale del documento d'identità di entrambi oltre alla copia fotostatica sottoscritta. Validità limitata alle sole spedizioni elencate.")
    section_b(p); p.sign("Torino, 21 novembre 2025")


def t06(p, r):
    p.title("CONTRATTO DI LOCAZIONE AD USO ABITATIVO"); p.title("Estratto dei dati essenziali", 46)
    p.para("Tra il locatore e il conduttore, come di seguito identificati, si conviene la locazione dell'appartamento al terzo piano con cantina pertinenziale, per una durata di quattro anni rinnovabili tacitamente.")
    p.line_field("Locatore", "Immobiliare Corsini Ghirardelli S.a.s.", "company")
    p.comb("Partita IVA locatore", partita_iva(r), "vat")
    p.line_field("Conduttore", "Lombardozzi Fiammetta", "name")
    p.comb("Codice fiscale conduttore", codice_fiscale(r, "Lombardozzi", "Fiammetta", 1990, 4, 17, 1, "A944"), "cf")
    p.comb_row([("Canone mensile", "785,00", "amount"), ("Decorrenza", D(1, 3, 2026), "date")])
    p.comb_row([("Foglio", "112", "number"), ("Particella", "4479", "number"), ("Subalterno", "23", "number")])
    p.ticks(["Canone libero", "Canone concordato", "Cedolare secca"], {1, 2})
    p.para("Deposito cauzionale corrispondente a tre mensilità, infruttifero. Spese ordinarie di portierato e ascensore ripartite secondo le tabelle millesimali allegate.")
    section_b(p); p.sign("Bologna, 12 febbraio 2026")


def t07(p, r):
    p.title("CONDOMINIO PARCO DEI GELSI"); p.title("Verbale di assemblea straordinaria", 48)
    p.para("Alle ore diciannove, in seconda convocazione, nel locale portineria, si sono riuniti i condòmini per deliberare sul rifacimento della facciata ammalorata e sulla sostituzione dell'impianto citofonico ormai obsoleto.")
    p.line_field("Amministratore", "Gestioni Pellegatta", "company")
    p.comb("Codice fiscale condominio", "97" + "".join(r.choice(string.digits) for _ in range(9)), "id")
    p.comb_row([("Millesimi presenti", "684", "number"), ("Votanti", "17", "number")])
    p.comb("Preventivo approvato euro", "96.300,00", "amount")
    p.comb("IBAN conto condominiale", iban_it(r), "iban")
    p.ticks(["Approvato all'unanimità", "Approvato a maggioranza", "Rinviato"], {1})
    p.para("Il geometra incaricato presenterà la domanda di permesso edilizio entro trenta giorni. Le rate straordinarie saranno esigibili a partire dal mese di aprile, con possibilità di rateizzazione trimestrale.")
    section_b(p); p.sign("Monza, 9 marzo 2026")


def t08(p, r):
    p.title("CONSTATAZIONE AMICHEVOLE DI INCIDENTE"); p.title("Veicolo A", 50)
    p.para("Tamponamento avvenuto all'incrocio semaforico tra viale Garibaldi e via Manin, con asfalto bagnato e visibilità ridotta dalla nebbia mattutina. Nessun ferito, danni al paraurti posteriore e al fanale sinistro.")
    p.line_field("Contraente", "Dall'Olio Settimio", "name")
    p.comb("Targa", "GH 417 TZ", "plate")
    p.comb("Codice fiscale", codice_fiscale(r, "Dallolio", "Settimio", 1958, 10, 2, 0, "D548"), "cf")
    p.comb_row([("Data sinistro", D(6, 1, 2026), "date"), ("Ora", "0745", "time")])
    p.comb("Numero polizza", "RCA7730218845", "id")
    p.comb("Telefono", P(r), "phone")
    p.ticks(["In sosta", "Ripartiva dopo l'arresto", "Cambiava corsia", "Non rispettava la precedenza"], {1})
    p.para("Testimone presente: un ciclista residente nella zona, disponibile a rilasciare dichiarazione scritta all'ispettore liquidatore.")
    section_b(p); p.sign("Ferrara, 6 gennaio 2026")


def t09(p, r):
    p.title("A.S.D. PALLAVOLO ROCCARASO"); p.title("Tesseramento atleta stagione agonistica", 46)
    p.para("Il genitore esercente la responsabilità chiede il tesseramento della figlia minorenne nella categoria under quattordici, accettando lo statuto sociale e il regolamento disciplinare deliberato dal consiglio direttivo.")
    p.line_field("Atleta", "Iannucci Viola", "name")
    p.comb("Codice fiscale atleta", codice_fiscale(r, "Iannucci", "Viola", 2012, 5, 11, 1, "H400"), "cf")
    p.comb_row([("Scadenza visita medica", D(30, 6, 2026), "date"), ("Taglia maglia", "XS", "size")])
    p.comb("Quota stagionale euro", "260,00", "amount")
    p.comb("Cellulare genitore", P(r), "phone")
    p.ticks(["Consenso alle riprese fotografiche", "Trasporto con pulmino sociale", "Allergie alimentari segnalate"], {0, 1})
    p.para("Allenamenti nella palestra comunale il martedì e il giovedì pomeriggio. Il certificato di idoneità non agonistica deve pervenire prima della prima partita di campionato.")
    section_b(p); p.sign("Roccaraso, 2 settembre 2025")


def t10(p, r):
    p.title("ENERGIA DOLOMITI"); p.title("Richiesta di nuova attivazione fornitura gas naturale", 44)
    p.para("Il richiedente domanda l'attivazione del contatore presso l'immobile ristrutturato, dichiarando che l'impianto domestico è conforme e corredato delle certificazioni rilasciate dall'installatore abilitato.")
    p.line_field("Intestatario", "Kofler Matthias", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Kofler", "Matthias", 1985, 2, 26, 0, "A952"), "cf")
    p.comb("Codice PDR", "".join(r.choice(string.digits) for _ in range(14)), "id")
    p.comb("Matricola contatore", "G4R0039281", "id")
    p.comb_row([("CAP", "39100", "cap"), ("Consumo annuo stimato mc", "1350", "number")])
    p.comb("IBAN per domiciliazione", iban_it(r), "iban")
    p.ticks(["Uso domestico", "Riscaldamento centralizzato", "Uso commerciale"], {0})
    p.para("La prima bolletta comprenderà i costi fissi di allacciamento. In caso di morosità, il distributore potrà procedere alla sospensione previo sollecito scritto via posta certificata.")
    section_b(p); p.sign("Bolzano, 19 ottobre 2025")


def x01(p, r):
    p.title("COMUNE DI SPOLETO - SETTORE EDILIZIA"); p.title("Istanza di contributo per recupero del centro storico", 44)
    p.para("Il proprietario chiede l'erogazione del contributo previsto dal bando per il consolidamento delle murature portanti e il restauro conservativo delle persiane lignee del palazzetto vincolato.")
    p.line_field("Richiedente", "Mancinelli Adalgisa", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Mancinelli", "Adalgisa", 1961, 8, 19, 1, "I921"), "cf")
    p.comb("Partita IVA impresa esecutrice", partita_iva(r), "vat")
    p.comb_row([("Spesa ammissibile euro", "58.920,00", "amount"), ("Protocollo", "20931", "number")])
    p.comb("IBAN per l'erogazione", iban_it(r), "iban")
    p.comb("Data fine lavori prevista", D(15, 10, 2026), "date")
    p.ticks(["Immobile soggetto a tutela", "Residenza del proprietario", "Locato a terzi"], {0, 1})
    p.para("La soprintendenza ha espresso parere favorevole con prescrizioni sulle tinteggiature a calce. Liquidazione subordinata al collaudo e alla presentazione delle fatture quietanzate.")
    section_b(p); p.sign("Spoleto, 27 aprile 2026")


def x02(p, r):
    p.title("TRENITALIA REGIONALE - SERVIZIO CLIENTI"); p.title("Reclamo per ritardo e indennizzo", 48)
    p.para("Il viaggiatore segnala un ritardo oltre novanta minuti sulla tratta tra Pescara e Ancona, con mancata coincidenza e pernottamento imprevisto, e chiede il ristoro previsto dalla carta dei servizi.")
    p.line_field("Viaggiatore", "Di Censo Ottavio", "name")
    p.comb("Codice fiscale", codice_fiscale(r, "Dicenso", "Ottavio", 1972, 11, 3, 0, "G482"), "cf")
    p.comb_row([("Numero treno", "4127", "number"), ("Data viaggio", D(23, 7, 2026), "date")])
    p.comb("Codice biglietto", "PNR8XQ27LM", "id")
    p.comb("Importo pagato euro", "38,90", "amount")
    p.comb("IBAN per il rimborso", iban_it(r), "iban")
    p.ticks(["Indennizzo in denaro", "Bonus elettronico", "Rimborso integrale"], {0})
    p.para("Si allegano lo scontrino dell'albergo e il giustificativo del taxi notturno. Il sottoscritto resta reperibile all'indirizzo di posta indicato nel profilo personale.")
    section_b(p); p.sign("Pescara, 24 luglio 2026")


SECTION_B = {
    "T01": ("Riquadro B - Componenti del nucleo", ["Nominativo", "Parentela", "Nascita"],
            [["Bertolini Lucrezia", "coniuge", "11/06/1983"], ["Bertolini Tommaso", "figlio", "02/09/2017"]],
            "L'ufficiale d'anagrafe effettuerà l'accertamento di dimora abituale entro quarantacinque giorni; in mancanza di riscontro la pratica si intende confermata per silenzio assenso."),
    "T02": ("Riepilogo addebiti", ["Voce", "Euro"], [["Importo disposto", "1.846,50"], ["Commissione", "1,75"], ["Totale addebitato", "1.848,25"]],
            "Conservare la contabile come prova liberatoria. Per contestazioni rivolgersi alla filiale entro tredici mesi dalla data valuta."),
    "T03": ("Carriera pregressa", ["Insegnamento", "Crediti", "Voto"], [["Letteratura greca", "12", "30"], ["Glottologia", "9", "28"], ["Paleografia latina", "6", "27"]],
            "La segreteria didattica comunicherà l'esito della convalida tramite il portale riservato; i crediti eccedenti potranno confluire nel piano didattico individuale."),
    "T04": ("Dettaglio fatture", ["Numero", "Prestazione", "Euro"], [["118/A", "Ablazione tartaro", "90,00"], ["131/A", "Ricostruzione molari", "520,00"], ["136/A", "Ortopantomografia", "132,00"]],
            "Le somme liquidate saranno accreditate entro sessanta giorni lavorativi, al netto dello scoperto contrattuale indicato nella tabella dei massimali."),
    "T05": ("Spedizioni oggetto della delega", ["Codice", "Mittente", "Tipologia"], [["RA779301245IT", "Agenzia Entrate", "raccomandata AR"], ["CP448120773IT", "Libreria Sabauda", "pacco"]],
            "Il personale addetto verificherà la corrispondenza delle firme apposte; eventuali abrasioni o correzioni rendono nullo il presente modulo."),
    "T06": ("Piano dei versamenti", ["Rata", "Scadenza", "Euro"], [["Marzo", "01/03/2026", "785,00"], ["Aprile", "01/04/2026", "785,00"], ["Cauzione", "28/02/2026", "2.355,00"]],
            "La registrazione telematica presso l'Agenzia competente avverrà a cura del locatore, che ne trasmetterà copia al conduttore entro due mesi."),
    "T07": ("Esito delle votazioni", ["Punto all'ordine del giorno", "Favorevoli", "Contrari"], [["Rifacimento facciata", "14", "3"], ["Nuovo citofono", "17", "0"], ["Fondo speciale", "12", "5"]],
            "Il presidente dichiara chiusa la seduta alle ventuno e trenta; copia del verbale verrà affissa nella bacheca dell'androne e spedita agli assenti."),
    "T08": ("Danni visibili", ["Parte", "Gravità", "Foto"], [["Paraurti posteriore", "media", "3"], ["Fanale sinistro", "lieve", "2"], ["Portellone", "graffio", "1"]],
            "Schizzo planimetrico disegnato sul retro. Entrambi i conducenti confermano la dinamica, sottoscrivendo ciascuno la propria copia carbone."),
    "T09": ("Calendario partite", ["Turno", "Avversaria", "Campo"], [["1", "Volley Sulmona", "casa"], ["2", "Polisportiva Castel di Sangro", "trasferta"], ["3", "Gioventù Pescocostanzo", "casa"]],
            "Le divise da gara restano di proprietà societaria e vanno riconsegnate lavate al termine del torneo primaverile."),
    "T10": ("Apparecchi collegati", ["Apparecchio", "Potenza kW", "Installazione"], [["Caldaia a condensazione", "24", "2025"], ["Piano cottura", "7", "2025"], ["Scaldabagno", "11", "2019"]],
            "Il tecnico distributore eseguirà la prova di tenuta prima della posa del misuratore; l'appuntamento verrà fissato con preavviso telefonico."),
    "X01": ("Quadro economico", ["Lavorazione", "Quantità", "Euro"], [["Iniezioni consolidanti", "48 mq", "21.600,00"], ["Restauro persiane", "14 pz", "9.800,00"], ["Ponteggio", "1 corpo", "27.520,00"]],
            "Il direttore dei lavori attesterà il regolare compimento; eventuali varianti in corso d'opera richiedono nuova autorizzazione paesaggistica."),
    "X02": ("Esborsi documentati", ["Descrizione", "Importo"], [["Pernottamento", "74,00"], ["Taxi", "22,50"], ["Cena", "18,00"]],
            "L'azienda risponderà entro un mese; in difetto, il passeggero potrà adire l'organismo di conciliazione paritetica gratuitamente."),
}


def section_b(p):
    if p.b_done:
        return
    p.b_done = True
    title, header, rows, text = SECTION_B[p.lines[0]]
    p.heading(title); p.table(header, rows); p.para(text)


STYLES = {   # every layout different: title placement, section headings, short fields one or two to a line, table before or after the tick boxes
    "T01": {"title": "center", "head": "plain", "pair": False, "table_first": False},
    "T02": {"title": "left", "head": "band", "pair": True, "table_first": True},
    "T03": {"title": "center", "head": "box", "pair": True, "table_first": False},
    "T04": {"title": "left", "head": "plain", "pair": False, "table_first": True},
    "T05": {"title": "center", "head": "band", "pair": False, "table_first": False},
    "T06": {"title": "left", "head": "box", "pair": False, "table_first": True},
    "T07": {"title": "center", "head": "plain", "pair": True, "table_first": True},
    "T08": {"title": "left", "head": "band", "pair": True, "table_first": False},
    "T09": {"title": "center", "head": "box", "pair": False, "table_first": True},
    "T10": {"title": "left", "head": "plain", "pair": True, "table_first": False},
    "X01": {"title": "left", "head": "box", "pair": True, "table_first": True},
    "X02": {"title": "center", "head": "band", "pair": False, "table_first": False},
}


DOCS = [("T01", "helv", t01), ("T02", "georgia", t02), ("T03", "times", t03), ("T04", "verdana", t04), ("T05", "trebuchet", t05),
        ("T06", "times", t06), ("T07", "helv", t07), ("T08", "verdana", t08), ("T09", "georgia", t09), ("T10", "trebuchet", t10),
        ("X01", "helv", x01), ("X02", "georgia", x02)]
SHARED = {"codice", "fiscale", "telefono", "cellulare", "importo", "indirizzo", "documento", "identità", "presente", "sottoscritto",
          "allegati", "firma", "richiesta", "richiedente", "domanda", "quietanzate", "fatture", "originale", "validità", "dichiara",
          "indicato", "indicati", "previsto", "prevista", "secondo", "presso", "entrambi", "attivazione", "erogazione", "mensile",
          "intestatario", "rimborso", "giorni", "eventuali", "saranno", "propria", "personale", "nascita", "totale", "cognome", "chiede", "comune", "d'identità", "dichiarazione", "matricola", "responsabilità",
          "gennaio", "febbraio", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre", "accredito", "contributo", "beneficiario", "partita", "numero", "scadenza", "allegato"}


def check_vocabulary(pages):
    words = {code: {w for w in re.findall(r"[a-zàèéìòóù']+", "\n".join(p.lines).lower()) for w in [w.strip("'")] if len(w) >= 6} for code, p in pages.items()}
    seen = {}
    for code, ws in words.items():
        for w in ws - SHARED:
            seen.setdefault(w, []).append(code)
    repeated = {w: c for w, c in seen.items() if len(c) > 1}
    assert not repeated, f"content words in more than one document: {sorted(repeated.items())}"
    every = set().union(*words.values())
    return len(every), sum(len(re.findall(r"\S+", "\n".join(p.lines))) for p in pages.values())


pages = {code: doc(code, face, 20261014 + k, build) for k, (code, face, build) in enumerate(DOCS)}
import inspect
for code, (_, _, build) in zip(pages, DOCS):   # every field the builder asks for is on the page and in the text
    src = inspect.getsource(build)
    asked = src.count("p.comb(") + src.count("p.line_field(") + sum(r.count('("') for r in re.findall(r"p\.comb_row\(\[(.*?)\]\)", src))
    have = [e for e in pages[code].entities if e["type"] != "table"]
    assert len(have) == asked, (code, asked, len(have))
    for e in have:
        assert any(e["text"] in line for line in pages[code].lines), (code, e)
distinct, total = check_vocabulary(pages)
meta = []
for code, p in pages.items():
    p.img.save(OUT / f"{code}.png")
    (OUT / f"{code}.txt").write_text("\n".join(p.lines) + "\n", encoding="utf-8")
    meta.append({"id": code, "gt": "generated", "level": 1, "tpl": "form", "lang": "it", "rot90": False, "size": [W, H],
                 "role": "train" if code.startswith("T") else "test", "entities": p.entities})
json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
first, *rest = [p.img for p in pages.values()]
first.save(OUT / "print.pdf", save_all=True, append_images=rest, resolution=300)
n = sum(len(m["entities"]) for m in meta); combs = sum(e["type"] in ("cf", "iban", "vat", "date", "cap", "phone", "amount", "id", "number", "plate", "time", "size") for m in meta for e in m["entities"])
print(f"{len(meta)} pages, {total} words, {distinct} distinct words of 6+ letters, {n} entities ({combs} in comb fields) -> {OUT}/print.pdf")
