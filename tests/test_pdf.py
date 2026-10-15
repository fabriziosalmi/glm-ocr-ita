"""glmocr/pdf.py: which pages go to the text layer and which to the model. No model is loaded: a stub reader stands in for it."""
import glob

import pytest
from PIL import Image, ImageDraw

from glmocr import pdf
from glmocr.corpus import ROOT


class Stub:
    def __init__(self):
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return {"text": "MODEL", "looped": False}


def test_bad_chars():
    assert pdf.bad_char_count("Fattura n. 12") == 0
    assert pdf.bad_char_count("�x") == 2


def test_alien_letters():
    """The Gazzetta's small caps come out as Malayalam and Sinhala; Greek in formulas and a page in another script are fine."""
    body = "Il Ministro dell'economia e delle finanze, visto il decreto legislativo " * 5
    assert pdf.alien_letters(body + "Gൺඓඓൾඍඍൺ Uൿൿංർංൺඅൾ") >= 3
    assert pdf.alien_letters(body + "con una concentrazione di 5 μg/l e un angolo α") == 0
    assert pdf.alien_letters(body + "5 µm, 3 µg, 2 µs e la variabile 𝑥 di 𝑓(𝑥) = 𝑒") == 0   # MICRO SIGN, mathematical italic
    assert pdf.alien_letters("这是一份中文文件 " * 20) == 0


def test_order_backtracks():
    down = [(0, y, 100, y + 10) for y in range(0, 200, 12)]
    two_columns = down + [(200, y, 300, y + 10) for y in range(0, 200, 12)]   # up to the top of the column on the right: fine
    assert pdf.order_backtracks(down) == 0 and pdf.order_backtracks(two_columns) == 0
    assert pdf.order_backtracks(down[::-1]) == len(down) - 1                   # bottom to top: every move goes up


def test_page_ranges():
    assert pdf.page_ranges("1-3,7", 10) == [0, 1, 2, 6]
    assert pdf.page_ranges(None, 3) == [0, 1, 2]
    assert pdf.page_ranges("9-12", 10) == [8, 9]


def test_scan_goes_to_the_model(tmp_path):
    """A PDF made of an image has no text layer: auto reads it with the model, text gives nothing, and the model is loaded only then."""
    im = Image.new("RGB", (1240, 1754), "white"); ImageDraw.Draw(im).text((100, 100), "Fattura n. 12 del 3 marzo 2025", fill="black")
    f = tmp_path / "scan.pdf"; im.save(f)
    made = []
    out = list(pdf.read_pdf(f, mode="auto", make_reader=lambda: made.append(Stub()) or made[-1]))
    assert [(r["source"], r["reason"], r["text"]) for r in out] == [("ocr", "no text layer", "MODEL")] and made[0].calls == 1
    assert [(r["source"], r["text"]) for r in pdf.read_pdf(f, mode="text", make_reader=lambda: pytest.fail("loaded"))] == [("text", "")]


@pytest.mark.skipif(not glob.glob(str(ROOT / "external/gu_src/*.pdf")), reason="Gazzetta PDFs not on this machine (generators/make_gu.py)")
def test_born_digital_page_is_read_from_its_layer():
    """A Gazzetta page is born digital: auto takes its layer, never loads the model, and keeps the words of the page."""
    f = sorted(glob.glob(str(ROOT / "external/gu_src/*.pdf")))[-1]   # the newest issue: its running head has a sound font map
    (r,) = pdf.read_pdf(f, mode="auto", pages="2", make_reader=lambda: pytest.fail("the model was loaded for a digital page"))
    assert r["source"] == "text" and r["reason"] == "" and "GAZZETTA UFFICIALE" in r["text"].upper() and r["s"] < 5
    ocr = Stub(); (o,) = pdf.read_pdf(f, reader=ocr, mode="ocr", pages="2")
    assert o["source"] == "ocr" and ocr.calls == 1
    old = sorted(glob.glob(str(ROOT / "external/gu_src/gu_2019-02-20_*.pdf")))   # small caps that come out as Malayalam
    if old:
        stub = Stub(); (b,) = pdf.read_pdf(old[0], reader=stub, mode="auto", pages="2")
        assert b["source"] == "ocr" and b["reason"].startswith("broken font map") and stub.calls == 1
