"""work/gen_venv/bin/python generators/make_f24.py [N] : N filled copies of the official F24 form (Agenzia delle Entrate, "Modello di
versamento unificato - F24 Ordinario", page 1; external/forms_src/f24_ordinario.pdf, docs/EXTERNAL.md) -> external/f24_synth/
{id}.png, {id}.txt, meta.json. Training data for comb fields: the form is the real one, the values are made up.
Runs in its own environment (work/gen_venv: pypdfium2, pillow), so the measuring environment is never touched.

The form is rendered at 300 dpi and the values are written into its boxes, whose positions come from the PDF's own vector
paths (in points, origin top left). The printed labels are LABELS below, transcribed once by hand from the PDF's text layer,
which has broken ligatures and split words ("f lli ... a mentare" for "fallimentare"). The rotated text in the left margin is
left out. Ground truth: labels and values together, grouped into lines by their vertical centre and read left to right, as the
model reads a form line by line. A comb field is its characters written together; an amount is written with its comma.
Entities: every value written."""
import json, os, pathlib, random, string, sys
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

ROOT = pathlib.Path(__file__).resolve().parent.parent; os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "generators"))
from ids import codice_fiscale, iban_it, COMUNI, SURNAMES, NAMES_M, NAMES_F, BANKS   # noqa: E402

SRC, OUT = pathlib.Path("external/forms_src/f24_ordinario.pdf"), pathlib.Path("external/f24_synth")
DPI = 300; K = DPI / 72
S = "/System/Library/Fonts/"; SS = S + "Supplemental/"
PENS = [(SS + "Bradley Hand Bold.ttf", (25, 35, 120)), (SS + "Courier New Bold.ttf", (20, 20, 20)), (S + "Helvetica.ttc", (20, 20, 20)),
        (SS + "Courier New.ttf", (30, 30, 30)), (SS + "Bradley Hand Bold.ttf", (20, 20, 20))]

# (top in points, x, text): the printed labels of page 1, in the PDF's own coordinates
LABELS = [(26, 523, "Mod. F24"), (33, 21, "Agenzia Entrate"), (50, 244, "DELEGA IRREVOCABILE A:"), (61, 21, "MODELLO DI PAGAMENTO"),
          (75, 21, "UNIFICATO"), (75, 299, "AGENZIA"), (75, 502, "PROV."), (86, 244, "PER L’ACCREDITO ALLA TESORERIA COMPETENTE"),
          (96, 25, "CONTRIBUENTE"), (108, 29, "CODICE FISCALE"), (108, 463, "barrare in caso di anno d’imposta"),
          (115, 466, "non coincidente con anno solare"), (122, 114, "cognome, denominazione o ragione sociale"), (122, 417, "nome"),
          (135, 29, "DATI ANAGRAFICI"), (146, 113, "data di nascita"), (146, 233, "sesso (M o F)"),
          (146, 273, "comune (o Stato estero) di nascita"), (146, 546, "prov."), (155, 121, "giorno"), (155, 152, "mese"), (155, 194, "anno"),
          (172, 114, "comune"), (172, 338, "prov."), (172, 365, "via e numero civico"), (183, 29, "DOMICILIO FISCALE"),
          (199, 29, "CODICE FISCALE del coobbligato, erede,"), (207, 29, "genitore, tutore o curatore fallimentare"),
          (207, 479, "codice identificativo"), (218, 25, "SEZIONE ERARIO"), (227, 218, "rateazione/regione/"), (227, 283, "anno di"),
          (232, 166, "codice tributo"), (232, 224, "prov./mese rif."), (232, 280, "riferimento"), (232, 331, "importi a debito versati"),
          (232, 410, "importi a credito compensati"), (254, 29, "IMPOSTE DIRETTE – IVA"), (266, 29, "RITENUTE ALLA FONTE"),
          (278, 29, "ALTRI TRIBUTI ED INTERESSI"), (302, 488, "+/– SALDO (A-B)"), (305, 23, "codice ufficio"), (305, 79, "codice atto"),
          (317, 260, "TOTALE A"), (317, 402, "B"), (327, 25, "SEZIONE INPS"),
          (337, 23, "codice"), (337, 57, "causale"), (337, 112, "matricola INPS/codice INPS/"), (337, 233, "periodo di riferimento:"),
          (342, 23, "sede"), (342, 56, "contributo"), (342, 132, "filiale azienda"), (342, 216, "da mm/aaaa"), (342, 269, "a mm/aaaa"),
          (342, 331, "importi a debito versati"), (342, 410, "importi a credito compensati"),
          (386, 488, "+/– SALDO (C-D)"), (401, 260, "TOTALE C"), (401, 402, "D"), (411, 25, "SEZIONE REGIONI"),
          (420, 23, "codice"), (420, 225, "rateazione/"), (420, 283, "anno di"),
          (425, 23, "regione"), (425, 166, "codice tributo"), (425, 228, "mese rif."), (425, 280, "riferimento"),
          (425, 331, "importi a debito versati"), (425, 410, "importi a credito compensati"),
          (470, 488, "+/– SALDO (E-F)"), (485, 260, "TOTALE E"), (485, 402, "F"),
          (495, 25, "SEZIONE IMU E ALTRI TRIBUTI LOCALI"), (495, 229, "IDENTIFICATIVO OPERAZIONE"),
          (504, 23, "codice ente/"), (504, 87, "Immob."), (504, 131, "numero"), (504, 225, "rateazione/"), (504, 283, "anno di"),
          (509, 23, "codice comune"), (509, 74, "Ravv."), (509, 87, "variati"), (509, 103, "Acc."), (509, 115, "Saldo"), (509, 131, "immobili"),
          (509, 166, "codice tributo"), (509, 232, "mese rif."), (509, 280, "riferimento"), (509, 331, "importi a debito versati"),
          (509, 410, "importi a credito compensati"), (554, 488, "+/– SALDO (G-H)"), (568, 38, "detrazione"), (568, 260, "TOTALE G"),
          (568, 402, "H"), (579, 25, "SEZIONE ALTRI ENTI PREVIDENZIALI E ASSICURATIVI"),
          (590, 101, "codice sede"), (590, 156, "codice ditta"), (590, 210, "c.c."), (590, 240, "numero di riferimento"), (590, 284, "causale"),
          (590, 331, "importi a debito versati"), (590, 410, "importi a credito compensati"), (614, 26, "INAIL"),
          (626, 488, "+/– SALDO (I-L)"), (640, 260, "TOTALE I"), (640, 403, "L"),
          (652, 25, "codice ente"), (652, 65, "codice sede"), (652, 108, "causale contributo"), (652, 155, "codice posizione"),
          (652, 216, "periodo di riferimento: da mm/aaaa a mm/aaaa"), (652, 331, "importi a debito versati"),
          (652, 410, "importi a credito compensati"), (674, 488, "+/– SALDO (M-N)"), (688, 260, "TOTALE M"), (688, 402, "N"),
          (699, 25, "FIRMA"), (699, 330, "SALDO FINALE"), (711, 463, "EURO +"),
          (735, 25, "ESTREMI DEL VERSAMENTO (DA COMPILARE A CURA DI BANCA/POSTE/AGENTE DELLA RISCOSSIONE)"),
          (747, 144, "CODICE BANCA/POSTE/AGENTE DELLA RISCOSSIONE"), (747, 336, "Pagamento effettuato con assegno"),
          (747, 494, "bancario/postale"), (752, 77, "DATA"), (760, 177, "AZIENDA"), (760, 255, "CAB/SPORTELLO"),
          (760, 494, "circolare/vaglia postale"), (764, 335, "n.ro"), (772, 34, "giorno"), (772, 66, "mese"), (772, 109, "anno"),
          (775, 335, "tratto / emesso su"), (785, 426, "cod. ABI"), (785, 517, "CAB"), (800, 28, "Autorizzo addebito su"),
          (808, 28, "conto corrente codice IBAN"), (808, 348, "firma"),
          (821, 200, "1ª COPIA PER LA BANCA/POSTE/AGENTE DELLA RISCOSSIONE")]
LABEL_H = 6.5   # labels are 6 to 9 points high: their centre is top + LABEL_H / 2

# boxes (x, top, width, height, cells): a comb has one box per character; cells=0 is free text
CF = (113.4, 105.6, 230.4, 14.4, 16); COGNOME = (113.4, 129.6, 295.2, 14.5, 0); NOME = (415.8, 129.6, 151.2, 14.4, 0)
GIORNO = (113.4, 153.6, 28.8, 14.4, 2); MESE = (142.2, 153.6, 28.8, 14.4, 2); ANNO = (171.0, 153.6, 57.6, 14.4, 4); SESSO = (243.1, 153.6, 14.4, 14.4, 1)
COMUNE_N = (271.8, 153.6, 258.3, 14.4, 0); PROV_N = (538.0, 153.6, 28.8, 14.4, 2)
COMUNE_D = (113.4, 177.6, 208.8, 14.4, 0); PROV_D = (329.4, 177.6, 28.8, 14.4, 2); VIA = (365.4, 177.6, 201.6, 14.4, 0)
CF2 = (185.4, 201.6, 230.4, 14.4, 16); CODID = (538.0, 201.6, 28.8, 14.4, 2)
DELEGA = (335.0, 42.0, 238.0, 18.0, 0); AGENZIA = (335.0, 68.0, 160.0, 18.0, 0); PROV_A = (530.0, 68.0, 40.0, 18.0, 0)
UFFICIO = (19.8, 312.5, 43.2, 12.0, 3); ATTO = (77.4, 312.5, 158.9, 12.0, 11)
IDOP = (315.0, 493.2, 258.8, 9.9, 18); IBAN = (137.2, 802.5, 180.0, 12.0, 25)
ERARIO_ROWS = [242.0 + 12 * k for k in range(6)]   # top of each of the six rows of the Erario section
COLS = {"tributo": (156.5, 57.3), "rateazione": (221.7, 42.5), "anno": (271.8, 42.9)}
DEBIT_COMMA, CREDIT_COMMA, SALDO_COMMA = 385.3, 472.0, 558.1   # x of the printed comma of each amount column
TOT_ROW, SALDO_FIN_ROW = 312.5, 708.0
TRIBUTI = [("1001", "rateazione", None), ("1040", None, None), ("4001", None, "anno"), ("4033", None, "anno"), ("4034", None, "anno"),
           ("6099", None, "anno"), ("1668", None, "anno"), ("3918", None, "anno"), ("3843", "rateazione", "anno"), ("3844", "rateazione", "anno"),
           ("1990", None, "anno"), ("8904", None, "anno"), ("1712", None, "anno"), ("1713", None, "anno"), ("3800", None, "anno"),
           ("3812", None, "anno"), ("3813", None, "anno"), ("1038", None, "anno"), ("1053", None, None), ("6031", None, "anno")]


def amount_text(cents):
    """2.345,67: Italian notation, as written on the form"""
    euros, c = divmod(cents, 100)
    return f"{euros:,}".replace(",", ".") + f",{c:02d}"


class Form:
    def __init__(self, base, rng):
        self.img = base.copy(); self.d = ImageDraw.Draw(self.img); self.rng = rng
        self.items = []; self.entities = []
        self.pen, self.ink = rng.choice(PENS)
        self.jit = rng.uniform(0.0, 1.2)   # how far a written character strays from the centre of its box, in points

    def font(self, pt):
        return ImageFont.truetype(self.pen, max(8, int(pt * K)), index=0)

    def put(self, x, y_mid, s, pt, anchor="lm"):
        f = self.font(pt); j = lambda: self.rng.uniform(-self.jit, self.jit) * K
        self.d.text((x * K + j(), y_mid * K + j()), s, font=f, fill=self.ink, anchor=anchor)

    def box(self, b, value, kind):
        x, top, w, h, cells = b
        mid = top + h / 2
        if cells:
            assert len(value) <= cells, (value, cells)
            cw = w / cells
            for i, ch in enumerate(value):
                self.put(x + cw * (i + 0.5), mid, ch, min(h, cw) * 0.75, "mm")
        else:
            self.put(x + 3, mid, value, h * 0.62)
        self.items.append((mid, x + w / 2, value)); self.entities.append({"type": kind, "text": value})   # placed at the centre of its box, after the small labels printed inside it

    def amount(self, comma_x, top, cents, kind, h=12.0):
        """integer part right-aligned against the printed comma, cents in the two boxes after it"""
        mid = top + h / 2; t = amount_text(cents); whole, dec = t.split(",")
        self.put(comma_x - 1.5, mid, whole, h * 0.7, "rm")
        self.put(comma_x + 4.7, mid, dec[0], h * 0.7, "mm"); self.put(comma_x + 11.8, mid, dec[1], h * 0.7, "mm")
        self.items.append((mid, comma_x - 40, t)); self.entities.append({"type": kind, "text": t})

    def text(self):
        """labels and values in reading order: grouped by vertical centre (within 3.2 points), each group read left to right"""
        runs = [(top + LABEL_H / 2, x, s) for top, x, s in LABELS] + self.items
        runs.sort()
        lines, cur, y0 = [], [], None
        for y, x, s in runs:
            if cur and y - y0 > 3.2:
                lines.append(cur); cur = []
            if not cur:
                y0 = y
            cur.append((x, s))
        lines.append(cur)
        return "\n".join(" ".join(s for _, s in sorted(line)) for line in lines) + "\n"


def person(rng):
    female = rng.random() < 0.5
    name = rng.choice(NAMES_F if female else NAMES_M); surname = rng.choice(SURNAMES)
    year, month, day = rng.randrange(1940, 2004), rng.randrange(1, 13), rng.randrange(1, 29)
    born = rng.choice(COMUNI); home = rng.choice(COMUNI)
    return dict(name=name, surname=surname, female=female, year=year, month=month, day=day, born=born, home=home,
                cf=codice_fiscale(rng, surname, name, year, month, day, female, born[2]))


def fill(base, rng):
    f = Form(base, rng); p = person(rng)
    bank, branch_city = rng.choice(BANKS), rng.choice(COMUNI)
    f.box(DELEGA, bank, "bank"); f.box(AGENZIA, branch_city[0].title(), "branch"); f.box(PROV_A, branch_city[1], "prov")
    f.box(CF, p["cf"], "cf"); f.box(COGNOME, p["surname"].upper(), "surname"); f.box(NOME, p["name"].upper(), "name")
    f.box(GIORNO, f"{p['day']:02d}", "day"); f.box(MESE, f"{p['month']:02d}", "month"); f.box(ANNO, str(p["year"]), "year")
    f.box(SESSO, "F" if p["female"] else "M", "sex"); f.box(COMUNE_N, p["born"][0], "city"); f.box(PROV_N, p["born"][1], "prov")
    f.box(COMUNE_D, p["home"][0], "city"); f.box(PROV_D, p["home"][1], "prov")
    f.box(VIA, f"{rng.choice(['VIA', 'VIALE', 'PIAZZA', 'CORSO'])} {rng.choice(SURNAMES).upper()} {rng.randrange(1, 180)}", "street")
    if rng.random() < 0.25:
        q = person(rng); f.box(CF2, q["cf"], "cf"); f.box(CODID, rng.choice(["02", "07", "10", "51"]), "code")
    rows = rng.sample(TRIBUTI, rng.randrange(1, 6)); debit = credit = 0
    for top, (code, rate, anno) in zip(ERARIO_ROWS, rows):
        f.box((COLS["tributo"][0], top, COLS["tributo"][1], 12.0, 0), code, "tax_code")
        if rate:
            f.box((COLS["rateazione"][0], top, COLS["rateazione"][1], 12.0, 0), f"{rng.randrange(1, 7):02d}{rng.choice(['06', '09', '12'])}", "instalment")
        if anno:
            f.box((COLS["anno"][0], top, COLS["anno"][1], 12.0, 0), str(rng.randrange(2019, 2027)), "year")
        cents = rng.randrange(500, 900000)
        if rng.random() < 0.15 and debit > cents:
            f.amount(CREDIT_COMMA, top, cents, "credit"); credit += cents
        else:
            f.amount(DEBIT_COMMA, top, cents, "debit"); debit += cents
    f.amount(DEBIT_COMMA, TOT_ROW, debit, "total")
    if credit:
        f.amount(CREDIT_COMMA, TOT_ROW, credit, "total")
    f.amount(SALDO_COMMA, TOT_ROW, debit - credit, "balance"); f.amount(SALDO_COMMA, SALDO_FIN_ROW, debit - credit, "balance")
    if rng.random() < 0.3:
        f.box(UFFICIO, "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(3)), "office")
        f.box(ATTO, "".join(rng.choice(string.digits) for _ in range(11)), "act")
    if rng.random() < 0.5:
        f.box(IBAN, iban_it(rng)[2:], "iban")   # "IT" is printed on the form; the text gets the whole IBAN below
        f.items[-1] = (f.items[-1][0], f.items[-1][1], "IT" + f.items[-1][2]); f.entities[-1]["text"] = "IT" + f.entities[-1]["text"]
    return f


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    OUT.mkdir(parents=True, exist_ok=True)
    base = pdfium.PdfDocument(str(SRC))[0].render(scale=K).to_pil().convert("RGB")
    meta = []
    for i in range(n):
        rng = random.Random(f"f24-{i}")
        f = fill(base, rng); pid = f"f24_{i:04d}"
        f.img.save(OUT / f"{pid}.png"); (OUT / f"{pid}.txt").write_text(f.text(), encoding="utf-8")
        meta.append({"id": pid, "gt": "generated", "level": 1, "tpl": "f24", "lang": "it", "rot90": False, "size": list(f.img.size),
                     "entities": f.entities})
    json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(n, "F24 forms,", sum(len(m["entities"]) for m in meta), "values ->", OUT)
