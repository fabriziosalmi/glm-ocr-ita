"""Generate synthetic it/en document pages + exact ground truth (eval/NN_lang_kind.{png,txt})."""
import random, json, pathlib
from PIL import Image, ImageDraw, ImageFont, ImageFilter
random.seed(7)
OUT = pathlib.Path(__file__).parent / "eval"
F = {"times": ("/System/Library/Fonts/Times.ttc", 0), "helv": ("/System/Library/Fonts/Helvetica.ttc", 0),
     "helvb": ("/System/Library/Fonts/Helvetica.ttc", 1), "timesb": ("/System/Library/Fonts/Times.ttc", 1)}
def font(k, s): return ImageFont.truetype(F[k][0], s, index=F[k][1])

IT = ["La società si impegna a fornire i servizi descritti nel presente contratto entro trenta giorni dalla data di sottoscrizione, salvo cause di forza maggiore debitamente documentate.",
 "Il Responsabile del trattamento è tenuto a garantire che le persone autorizzate abbiano assunto un impegno di riservatezza e adottare tutte le misure di sicurezza richieste dall'articolo 32.",
 "Nel corso dell'esercizio 2025 il fatturato è cresciuto del 12,4% rispetto all'anno precedente, grazie soprattutto all'espansione nei mercati dell'Europa meridionale e a un più efficiente utilizzo della logistica.",
 "Qualora il cliente non provveda al pagamento entro la scadenza indicata, decorreranno automaticamente gli interessi di mora nella misura prevista dal D.Lgs. 231/2002, oltre alle spese di recupero.",
 "La riunione del consiglio di amministrazione si è tenuta presso la sede di Genova il 14 marzo; è stata approvata all'unanimità la proposta di destinare l'utile a riserva straordinaria.",
 "Per qualsiasi controversia derivante dall'interpretazione o dall'esecuzione del presente accordo sarà competente in via esclusiva il Foro di Milano, con espressa rinuncia a ogni altro foro.",
 "Perché l'intervento sia efficace, è necessario verificare più volte la tensione dell'impianto: così si evitano guasti, perdite e—soprattutto—rischi per l'incolumità degli operatori."]
EN = ["The Supplier shall deliver the services described in this Agreement within thirty days of the effective date, save for force majeure events duly documented in writing.",
 "The Processor shall ensure that persons authorised to process the personal data have committed themselves to confidentiality and shall implement all measures required pursuant to Article 32.",
 "During fiscal year 2025 revenue grew by 12.4% compared with the previous year, driven mainly by expansion in Southern European markets and more efficient use of logistics.",
 "If the Customer fails to pay by the due date stated on the invoice, late-payment interest shall accrue automatically at the statutory rate, together with reasonable recovery costs.",
 "The board meeting was held at the Geneva headquarters on 14 March; the proposal to allocate the profit to an extraordinary reserve was approved unanimously by all directors present.",
 "Any dispute arising out of the interpretation or performance of this agreement shall be subject to the exclusive jurisdiction of the courts of London, to the exclusion of any other forum.",
 "To ensure the intervention is effective, the system voltage must be checked repeatedly: this prevents failures, leaks and—above all—risks to the safety of the operators."]
TITLES = {"it": ["Contratto di fornitura di servizi", "Verbale del consiglio di amministrazione", "Relazione annuale sulla gestione"],
          "en": ["Master Services Agreement", "Minutes of the Board Meeting", "Annual Management Report"]}
ROWS = {"it": [["Descrizione","Q.tà","Prezzo","Totale"],["Consulenza tecnica","10","€ 85,00","€ 850,00"],["Licenza annuale","2","€ 1.200,00","€ 2.400,00"],["Trasferta Genova–Milano","1","€ 143,50","€ 143,50"],["Totale imponibile","","","€ 3.393,50"]],
        "en": [["Description","Qty","Price","Total"],["Technical consulting","10","$ 85.00","$ 850.00"],["Annual licence","2","$ 1,200.00","$ 2,400.00"],["Travel London–Paris","1","$ 143.50","$ 143.50"],["Subtotal","","","$ 3,393.50"]]}

def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur+" "+w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]

def page(lang, kind, idx):
    W, H = 1240, 1754  # A4 @150dpi
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    paras = random.sample(IT if lang == "it" else EN, 4)
    gt = []; y = 120
    title = random.choice(TITLES[lang]); ft = font("helvb", 44)
    d.text((100, y), title, font=ft, fill="black"); gt.append(title); y += 100
    body = {"prose": ("times", 30), "dense": ("helv", 22), "twocol": ("times", 26), "table": ("helv", 28)}[kind]
    f = font(*body); lh = int(body[1]*1.45)
    if kind == "twocol":
        cw = (W-200-60)//2
        for c in range(2):
            yy = y
            for p in paras[2*c:2*c+2]:
                for ln in wrap(d, p, f, cw): d.text((100+c*(cw+60), yy), ln, font=f, fill="black"); yy += lh
                yy += lh//2
        gt += paras
    else:
        np_ = 6 if kind == "dense" else 3
        ps = (paras*2)[:np_]
        for p in ps:
            for ln in wrap(d, p, f, W-200): d.text((100, y), ln, font=f, fill="black"); y += lh
            y += lh//2
        gt += ps
    if kind == "table":
        rows = ROWS[lang]; cx = [100, 560, 700, 940, 1140]; y += 30
        for r, row in enumerate(rows):
            d.rectangle([100, y, 1140, y+56], outline="black", width=2)
            for c, cell in enumerate(row):
                d.text((cx[c]+12, y+12), cell, font=font("helvb" if r == 0 else "helv", 28), fill="black")
                if c: d.line([cx[c], y, cx[c], y+56], fill="black", width=2)
            gt += [c for c in row if c]; y += 56
    # scan-like degradation for odd pages
    if idx % 2:
        img = img.rotate(random.uniform(-0.8, 0.8), fillcolor="white", resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.6))
    name = f"{idx:02d}_{lang}_{kind}"
    img.save(OUT/f"{name}.png"); (OUT/f"{name}.txt").write_text("\n".join(gt), encoding="utf-8")

i = 0
for lang in ("it", "en"):
    for kind in ("prose", "dense", "twocol", "table"):
        page(lang, kind, i); i += 1
print(i, "pages")
