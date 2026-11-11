"""experiments/pieces.py, round 2: what a band returns that is not text of the page is not merged into the whole page."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pieces", Path(__file__).parent.parent / "experiments" / "pieces.py")
pieces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pieces)

WHOLE = "Spett.le Ditta Rossi\nOggetto: sollecito di pagamento\nCon la presente vi ricordiamo la fattura n. 118 del 3 marzo 2026."
NOISE = ["```markdown", "This image is too blurry to recognize any text.",
         "The image provided is too blurry to accurately recognize any text. It appears to be a close-up of a textured surface.",
         "This is a sample image of a text document. The text is not clearly visible.",
         "1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12.", "L'immagine non contiene testo leggibile."]
TEXT = ["Comune di Genova - Ufficio Tributi", "Distinti saluti, il Responsabile del procedimento", "Pag. 1 di 3"]


def band(*ls):
    return {"top": {"text": "\n".join(ls)}, "bottom": {"text": ""}}


def test_noise_is_not_merged_when_clean():
    for l in NOISE:
        assert l not in pieces.merge(WHOLE, band(l), use=("top",), clean=True).split("\n"), l


def test_noise_was_merged_before_round_2():
    merged = [l for l in NOISE if l in pieces.merge(WHOLE, band(l), use=("top",)).split("\n")]
    assert len(merged) >= 4, merged


def test_text_of_the_page_still_comes_in():
    for l in TEXT:
        assert l in pieces.merge(WHOLE, band("```markdown", l, "```"), use=("top",), clean=True).split("\n"), l


def test_counting_needs_five_numbers_in_a_row():
    assert pieces.counting("1. 2. 3. 4. 5.") and not pieces.counting("12 03 2026") and not pieces.counting("1. 2. 3.")
    assert not pieces.counting("2. 4. 6. 8. 10.")
