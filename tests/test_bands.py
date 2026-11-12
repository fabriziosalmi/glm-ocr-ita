"""glmocr/bands.py and glmocr/orient.py: the reader reads the page, then its two bands, and merges them in, unless the
orientation classifier is sure the page is turned."""

import numpy as np
import pytest
from PIL import Image

from glmocr import bands, corpus, orient

WHOLE = "Spett.le Ditta Rossi\nOggetto: sollecito di pagamento\nCon la presente vi ricordiamo la fattura n. 118 del 3 marzo 2026."
CLOSING = "Distinti saluti, il Responsabile del procedimento"


class FakeReader:
    """Answers by the size of what it is given: the whole page, or one of its bands."""
    def __init__(self, page):
        self.page, self.sizes, self.load_s = page, [], 1.5

    def read(self, image, scale=1.0, **kw):
        size = (image if isinstance(image, Image.Image) else Image.open(image)).size
        self.sizes.append(size)
        text = WHOLE if size == self.page.size else (CLOSING if size[1] < 0.15 * self.page.height else "```markdown\nSpett.le Ditta Rossi")
        self.last = {"text": text, "s": 1.0, "prompt_tokens": 10, "gen_tokens": 5, "loop_cut": None, "peak_gb": 1.0}
        return dict(self.last)


@pytest.fixture
def page():
    return Image.new("RGB", (1240, 1754), "white")


def test_bands_are_read_and_merged(page, monkeypatch):
    monkeypatch.setattr(orient, "classify", lambda im: (0, 0.93))
    r = bands.BandReader(FakeReader(page))
    out = r.read(page)
    assert r.reader.sizes == [(1240, 1754), (1240, round(0.16 * 1754)), (1240, 1754 - round(0.86 * 1754))]
    assert CLOSING in out["text"].split("\n") and out["whole_text"] == WHOLE
    assert "```markdown" not in out["text"]                     # a band's fence is not text of the page
    assert out["text"].count("Spett.le Ditta Rossi") == 1       # nor is a line the page already holds
    assert out["s"] >= 3.0 and r.load_s == 1.5                  # three reads; the inner reader's fields show through


def test_no_bands_on_a_page_the_classifier_is_sure_is_turned(page, monkeypatch):
    monkeypatch.setattr(orient, "classify", lambda im: (270, 0.92))
    r = bands.BandReader(FakeReader(page))
    out = r.read(page)
    assert len(r.reader.sizes) == 1 and out["text"] == WHOLE and out["bands"]["skipped"] == "turned"


def test_a_turn_the_classifier_is_unsure_of_does_not_cost_the_bands(page, monkeypatch):
    monkeypatch.setattr(orient, "classify", lambda im: (180, orient.SURE - 0.01))
    r = bands.BandReader(FakeReader(page))
    assert CLOSING in r.read(page)["text"] and len(r.reader.sizes) == 3


def test_resize_is_bilinear_without_antialiasing():
    """PaddleX's resize (cv2.INTER_LINEAR): a 1-pixel-wide line halved stays at half its grey, it is not spread out."""
    x = np.ones((512, 512, 3), np.float32); x[:, 255] = 0.0
    y = orient._resize_crop(x)                 # 512 -> 256, then the centre 224
    assert y.shape == (224, 224, 3)
    col = y[0, :, 0]
    assert col.min() == pytest.approx(0.5) and (col < 1).sum() == 1


def cached_weights():
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(orient.REPO, "model.safetensors", revision=orient.REVISION)
    return isinstance(p, str)


@pytest.mark.skipif(not cached_weights(), reason="the classifier's weights are not in the Hugging Face cache")
def test_the_classifier_finds_the_turned_pages_of_hard():
    try:
        d = corpus.corpus_dir("hard")
    except corpus.Missing:
        pytest.skip("hard/ is not on this machine")
    meta = corpus.load_meta("hard")
    answers = {m["id"]: orient.classify(Image.open(d / f"{m['id']}.png")) for m in meta}
    assert all((answers[m["id"]][0] != 0) == bool(m.get("rot90")) for m in meta)   # 40 of 40 right (2026-11-12)
    assert {m["id"] for m in meta if orient.turned(Image.open(d / f"{m['id']}.png"))} <= {m["id"] for m in meta if m.get("rot90")}
