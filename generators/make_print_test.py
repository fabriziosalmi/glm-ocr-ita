"""uv run --with reportlab python generators/make_print_test.py [--pdf PATH] : a two-page A4 test document to print, photograph and
read, with exact ground truth -> the PDF (default ~/Desktop/prova_ocr_stampa.pdf) and work/print_test/: page1.txt, page2.txt and
meta.json (entities with the block they sit in, so a photograph is scored block by block).

Every block is one of the conditions this repository found hard (docs/ROADMAP.md): a letterhead and a closing block after a large
blank space with a footer in small print under a rule (lines a whole-page reading drops), a dense two-column section in small print
(skipped blocks), comb fields, tick boxes and dot leaders, a table of near-identical rows (loops), amounts of tens of millions in
Italian notation, footnotes, accents, an English paragraph. All names, codes and companies are made up; tax code, VAT number and
IBAN carry valid check characters (generators/ids.py). Standard PDF fonts only (Helvetica, Times, Courier: euro sign and accents
included), so the text is exact and the PDF prints sharp. Reproducible: no clock, fixed seed."""
import argparse, json, pathlib, random, sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generators"))
from ids import codice_fiscale, iban_it, partita_iva  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--pdf", default=str(pathlib.Path.home() / "Desktop/prova_ocr_stampa.pdf"))
a = ap.parse_args()
rng = random.Random(20261111)
W, H = A4; M = 48                     # page size in points, margin
OUT = ROOT / "work/print_test"; OUT.mkdir(parents=True, exist_ok=True)

VAT, VAT2 = partita_iva(rng), partita_iva(rng)
IBAN, IBAN2 = iban_it(rng), iban_it(rng)
CF = codice_fiscale(rng, "Rossetti", "Paola", 1980, 1, 1, True, "D969")
CF2 = codice_fiscale(rng, "Ferri", "Marco", 1975, 6, 14, False, "F205")
group4 = lambda s: " ".join(s[i:i + 4] for i in range(0, len(s), 4))

pages, meta = [[], []], []
drawn = [[], []]   # every line as printed, to check that no entity is broken across two
ents = [[], []]


def ent(page, typ, text, block):
    ents[page].append({"type": typ, "text": text, "block": block})
    return text


class P:
    def __init__(self, c, page):
        self.c, self.page = c, page

    def line(self, x, y, s, font="Helvetica", size=10, block="body", right=False, gray=0.0):
        self.c.setFont(font, size); self.c.setFillGray(gray)
        (self.c.drawRightString if right else self.c.drawString)(x, y, s)
        pages[self.page].append((block, s)); drawn[self.page].append(s)

    def para(self, x, y, width, s, font="Helvetica", size=10, leading=None, block="body"):
        """wrapped on the page, one line in the ground truth; returns the y under it"""
        leading = leading or size * 1.25
        self.c.setFont(font, size); self.c.setFillGray(0)
        for ln in simpleSplit(s, font, size, width):
            self.c.drawString(x, y, ln); y -= leading; drawn[self.page].append(ln)
        pages[self.page].append((block, s))
        return y


def page1(c):
    p = P(c, 0); y = H - M
    # A. letterhead
    p.line(M, y - 14, "STUDIO LEGALE ROSSETTI & FERRI", "Times-Bold", 17, "top")
    p.line(M, y - 30, "Avvocati associati - Via XX Settembre 41/3, 16121 Genova", "Times-Roman", 9.5, "top")
    p.line(M, y - 43, f"P.IVA {ent(0, 'vat', VAT, 'top')} - Tel. 010 5550 1234", "Times-Roman", 9.5, "top")
    p.line(M, y - 56, f"PEC: {ent(0, 'pec', 'studio.rossettiferri@pec.example', 'top')}", "Times-Roman", 9.5, "top")
    p.line(W - M, y - 14, f"Prot. n. {ent(0, 'contract', '2291/2025', 'top')}", "Helvetica", 9, "top", right=True)
    p.line(W - M, y - 28, f"Genova, {ent(0, 'date', '11 novembre 2025', 'top')}", "Helvetica", 9, "top", right=True)
    c.setLineWidth(0.8); c.line(M, y - 66, W - M, y - 66)
    # B. recipient and body
    y -= 92
    for s in ["Spett.le Cantieri Navali Tirreno S.p.A.", "Alla c.a. del Direttore Amministrativo", "Calata Gadda 8, 16126 Genova"]:
        p.line(W / 2 + 20, y, s, "Helvetica", 10, "body"); y -= 13
    y -= 12
    y = p.para(M, y, W - 2 * M, "Oggetto: messa in mora e richiesta di pagamento ai sensi dell'art. 1219 c.c. - contratto "
               f"{ent(0, 'contract', 'CTR/2024/00417', 'body')} del {ent(0, 'date', '03/07/2024', 'body')}.", "Helvetica-Bold", 10)
    y -= 6
    y = p.para(M, y, W - 2 * M, "Egregi Signori, in nome e per conto della nostra assistita, Officine Meccaniche Liguri S.r.l., vi "
               f"comunichiamo che alla data odierna risulta ancora insoluto l'importo di {ent(0, 'amount', '1.234.567,89 €', 'body')}, "
               f"oltre agli interessi di mora al tasso del {ent(0, 'percent', '8,15%', 'body')} maturati dalla scadenza del "
               f"{ent(0, 'date', '31.03.2025', 'body')}. Poiché le precedenti sollecitazioni non hanno avuto esito, vi invitiamo a "
               "provvedere al saldo entro quindici giorni dal ricevimento della presente, più le spese indicate qui sotto.", "Times-Roman", 11)
    y -= 6
    y = p.para(M, y, W - 2 * M, f"Il pagamento andrà effettuato mediante bonifico sul conto IBAN {ent(0, 'iban', group4(IBAN), 'body')}, "
               "indicando nella causale il numero di protocollo. Decorso inutilmente il termine, procederemo senz'altro avviso.", "Times-Roman", 11)
    # C. fee table
    y -= 14
    cols = [M, M + 250, M + 320, W - M]
    for x, h in zip(cols, ["Voce", "Q.tà", "Prezzo unit.", "Importo"]):
        p.line(x, y, h, "Helvetica-Bold", 9, "table", right=(x == W - M))
    c.setLineWidth(0.5); c.line(M, y - 4, W - M, y - 4); y -= 16
    rows = [("Diritti di messa in mora", 1, 450.00), ("Esame documentazione contrattuale", 3, 180.00), ("Corrispondenza e PEC", 6, 25.50),
            ("Accesso agli atti", 2, 87.30), ("Spese generali forfettarie 15%", 1, 162.47)]
    total = 0.0
    it = lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    for d, q, u in rows:
        amt = q * u; total += amt
        p.line(cols[0], y, d, "Helvetica", 9, "table"); p.line(cols[1], y, str(q), "Helvetica", 9, "table")
        p.line(cols[2], y, it(u), "Helvetica", 9, "table"); p.line(cols[3], y, ent(0, "amount", f"{it(amt)} €", "table"), "Helvetica", 9, "table", right=True)
        y -= 13
    iva = round(total * 0.22, 2)
    c.line(M + 250, y + 6, W - M, y + 6)
    p.line(cols[2], y - 4, "Imponibile", "Helvetica", 9, "table"); p.line(cols[3], y - 4, ent(0, "amount", f"{it(total)} €", "table"), "Helvetica", 9, "table", right=True)
    p.line(cols[2], y - 17, "IVA 22%", "Helvetica", 9, "table"); p.line(cols[3], y - 17, ent(0, "amount", f"{it(iva)} €", "table"), "Helvetica", 9, "table", right=True)
    p.line(cols[2], y - 30, "Totale spese", "Helvetica-Bold", 9, "table"); p.line(cols[3], y - 30, ent(0, "amount", f"{it(total + iva)} €", "table"), "Helvetica-Bold", 9, "table", right=True)
    # D. large blank space, then E. closing block in the bottom fifth
    y = 190
    p.line(M, y, "Letto, confermato e sottoscritto.", "Times-Italic", 11, "closing")
    p.line(M, y - 16, f"Genova, {ent(0, 'date', '12 novembre 2025', 'closing')}", "Times-Roman", 11, "closing")
    for x, name, role in [(M, ent(0, "name", "Avv. Paola Rossetti", "closing"), "Socia fondatrice"),
                          (W / 2 + 20, ent(0, "name", "Dott. Marco Ferri", "closing"), "Praticante abilitato")]:
        c.setLineWidth(0.4); c.line(x, y - 58, x + 170, y - 58)
        p.line(x, y - 70, name, "Times-Roman", 10.5, "closing"); p.line(x, y - 82, role, "Times-Italic", 9, "closing")
    p.line(M, y - 104, "Documento firmato digitalmente ai sensi del D.Lgs. 82/2005 e norme collegate.", "Helvetica", 7.5, "closing")
    # F. footer in small print under a rule
    c.setLineWidth(0.6); c.line(M, 62, W - M, 62)
    p.line(M, 52, f"Studio Legale Rossetti & Ferri - C.F. {ent(0, 'cf', CF, 'footer')} - REA GE-{ent(0, 'contract', '478213', 'footer')}"
           f" - Capitale sociale {ent(0, 'amount', '€ 10.000,00', 'footer')} i.v.", "Helvetica", 6.5, "footer")
    p.line(M, 43, f"Conto dedicato: IBAN {ent(0, 'iban', group4(IBAN2), 'footer')} - Assicurazione RC professionale n. "
           f"{ent(0, 'contract', 'POL-2025-88412', 'footer')}", "Helvetica", 6.5, "footer")
    p.line(W - M, 30, "Pagina 1 di 2", "Helvetica", 6.5, "footer", right=True)


def page2(c):
    p = P(c, 1); y = H - M
    # G. form with comb fields, tick boxes and blanks
    p.line(M, y - 14, "MODULO DI RICHIESTA RIMBORSO SPESE", "Helvetica-Bold", 13, "form")
    p.line(M, y - 28, "Da compilare in stampatello e restituire firmato entro il termine indicato.", "Helvetica", 8, "form")
    y -= 52

    def comb(label, value, y):
        p.line(M, y + 16, label, "Helvetica", 7, "form", gray=0.35)
        bw = 15
        for k, ch in enumerate(value):
            c.setLineWidth(0.5); c.setStrokeGray(0.3); c.rect(M + k * bw, y - 4, bw, 15)
            c.setFont("Courier-Bold", 11); c.setFillGray(0); c.drawCentredString(M + k * bw + bw / 2, y, ch)
        c.setStrokeGray(0)
        pages[1].append(("form", value)); drawn[1].append(value)
        return y - 38
    y = comb("CODICE FISCALE DEL RICHIEDENTE", ent(1, "cf", CF2, "form"), y)
    y = comb("IBAN DI ACCREDITO", ent(1, "iban", iban_it(rng), "form"), y)
    p.line(M, y + 8, "MODALITÀ DI PAGAMENTO", "Helvetica", 7, "form", gray=0.35)
    for k, (lab, on) in enumerate([("Bonifico bancario", True), ("Assegno circolare", False), ("Compensazione F24", False)]):
        x = M + k * 160
        c.setLineWidth(0.6); c.rect(x, y - 8, 10, 10)
        if on:
            c.line(x + 1, y - 7, x + 9, y + 1); c.line(x + 1, y + 1, x + 9, y - 7)
        p.line(x + 15, y - 6, lab, "Helvetica", 9, "form")
    y -= 30
    p.line(M, y, "Importo richiesto: " + ent(1, "amount", "€ 3.418,72", "form"), "Helvetica", 10, "form")
    dots = "Luogo e data " + "." * 46 + "   Firma " + "." * 52
    p.line(M, y - 20, dots, "Helvetica", 9, "form")
    p.line(M, y - 38, "Note: " + "_" * 92, "Helvetica", 9, "form")
    # H. dense two columns in small print
    y -= 66
    c.setLineWidth(0.8); c.line(M, y + 12, W - M, y + 12)
    p.line(M, y, "RASSEGNA - APPALTI E FINANZIAMENTI PUBBLICI IN LIGURIA", "Times-Bold", 11, "columns")
    y -= 16
    cw = (W - 2 * M - 18) / 2
    cig, cup = "Z4A3B19C7E", "J31B24000150006"
    paras = [
        f"Il Comune di La Spezia ha aggiudicato il servizio di manutenzione delle aree verdi (CIG {ent(1, 'cig', cig, 'columns')}) "
        f"per un importo di {ent(1, 'amount', '2.346.810,00 €', 'columns')}, con un ribasso del {ent(1, 'percent', '12,37%', 'columns')} "
        "sulla base d'asta. L'avvio dei lavori è previsto entro la fine del trimestre.",
        f"La Regione ha finanziato la messa in sicurezza della litoranea con il progetto CUP {ent(1, 'cup', cup, 'columns')}: "
        f"lo stanziamento complessivo ammonta a {ent(1, 'amount', '29.467.745,94 €', 'columns')}, di cui una quota a carico "
        "del fondo per lo sviluppo e la coesione.",
        f"Il bando per le imprese artigiane chiude il {ent(1, 'date', '15 gennaio 2026', 'columns')}. Le domande vanno inviate "
        f"esclusivamente via PEC all'indirizzo {ent(1, 'pec', 'bandi.artigianato@pec.example', 'columns')}, allegando la visura "
        "camerale aggiornata e il documento unico di regolarità contributiva.",
        f"Già pubblicata la graduatoria provvisoria: le imprese ammesse sono 214, quelle escluse 37. Il contributo medio è di "
        f"{ent(1, 'amount', '18.250,00 €', 'columns')} e potrà essere erogato in due tranche, l'ultima entro il "
        f"{ent(1, 'date', '30/09/2026', 'columns')}.",
        "Più di un terzo delle domande respinte riguardava la mancata indicazione del codice ATECO o una firma digitale non "
        f"valida; il ricorso amministrativo è ammesso entro 60 giorni, come previsto dal {ent(1, 'contract', 'D.Lgs. 36/2023', 'columns')}.",
        f"A Savona l'Autorità di sistema portuale ha approvato il bilancio consuntivo con un avanzo di {ent(1, 'amount', '4.118.902,37 €', 'columns')}; "
        f"il collegio dei revisori ha espresso parere favorevole nella seduta del {ent(1, 'date', '21 ottobre 2025', 'columns')}, "
        "raccomandando però una revisione dei criteri di accantonamento per i contenziosi in corso.",
        f"Nel frattempo la Camera di commercio segnala che le nuove iscrizioni al registro delle imprese sono cresciute del "
        f"{ent(1, 'percent', '3,8%', 'columns')} rispetto all'anno precedente, trainate dai servizi alla persona e dal turismo "
        "costiero, mentre il manifatturiero resta stabile e l'edilizia rallenta dopo la fine degli incentivi.",
        f"Infine, il consorzio di bonifica ha pubblicato l'avviso per la manutenzione dei canali (CIG {ent(1, 'cig', 'B8D2F40A61', 'columns')}): "
        f"importo a base di gara {ent(1, 'amount', '687.430,15 €', 'columns')}, termine per le offerte alle ore 12:00 del "
        f"{ent(1, 'date', '09.02.2026', 'columns')}, sopralluogo obbligatorio previo appuntamento con l'ufficio tecnico.",
    ]
    lines = [ln for s_ in paras for ln in simpleSplit(s_, "Times-Roman", 7.5, cw) + [""]][:-1]
    half = (len(lines) + 1) // 2
    c.setFont("Times-Roman", 7.5); c.setFillGray(0)
    for k, ln in enumerate(lines):
        col, row = divmod(k, half)
        if ln:
            c.drawString(M + col * (cw + 18), y - row * 9.2, ln); drawn[1].append(ln)
    for s_ in paras:
        pages[1].append(("columns", s_))
    y -= half * 9.2
    # I. near-identical rows
    y -= 22
    p.line(M, y, "PIANO DI RATEIZZAZIONE", "Helvetica-Bold", 9, "rows")
    y -= 13
    for k in range(10):
        date = f"{[31, 28, 31, 30, 31, 30, 31, 31, 30, 31][k]:02d}/{k + 1:02d}/2026"
        p.line(M, y, f"Rata n. {k + 1} - scadenza {ent(1, 'date', date, 'rows')} - importo 1.250,00 € - stato: da pagare", "Courier", 8, "rows")
        y -= 10.5
    # J. English paragraph
    y -= 8
    y = p.para(M, y, W - 2 * M, f"Note for foreign applicants: from {ent(1, 'date', 'July 3, 2026', 'english')} every refund above "
               f"{ent(1, 'amount', '$ 12,450.00', 'english')} requires a certified translation of each invoice, and late requests are "
               "processed in the following quarter.", "Helvetica", 8.5, block="english")
    # K. footnotes in small print, then the footer
    c.setLineWidth(0.5); c.line(M, 92, M + 160, 92)
    notes = [f"(1) Delibera di Giunta regionale n. {ent(1, 'contract', '1187/2025', 'footnotes')}, pubblicata sul BURL del {ent(1, 'date', '04.12.2025', 'footnotes')}.",
             f"(2) Ai sensi dell'art. 2 del Regolamento, l'importo massimo per impresa è pari a {ent(1, 'amount', '75.000,00 €', 'footnotes')}.",
             f"(3) Responsabile del procedimento: {ent(1, 'name', 'Dott.ssa Elena Marchetti', 'footnotes')}, tel. 010 548 7733."]
    for k, s in enumerate(notes):
        p.line(M, 82 - k * 9, s, "Helvetica", 6, "footnotes")
    p.line(W - M, 30, "Pagina 2 di 2", "Helvetica", 6.5, "footer", right=True)


c = canvas.Canvas(a.pdf, pagesize=A4, invariant=1)
c.setTitle("Prova OCR"); c.setAuthor("glm-ocr-ita")
page1(c); c.showPage(); page2(c); c.showPage(); c.save()
for k in (0, 1):
    txt = "\n".join(s for _, s in pages[k])
    (OUT / f"page{k + 1}.txt").write_text(txt + "\n", encoding="utf-8")
    for e in ents[k]:
        assert e["text"] in txt, e
        assert any(e["text"] in ln for ln in drawn[k]), f"entity broken across printed lines: {e}"
    meta.append({"id": f"page{k + 1}", "gt": "exact", "level": 1, "tpl": ("letter", "form_columns")[k], "lang": "it", "rot90": False,
                 "blocks": list(dict.fromkeys(b for b, _ in pages[k])), "entities": ents[k]})
json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(a.pdf, "|", " ".join(f"page{k + 1}: {len(ents[k])} entities in {len(meta[k]['blocks'])} blocks" for k in (0, 1)), "->", OUT)
