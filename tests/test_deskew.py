"""glmocr/deskew.py on a drawn page turned by known angles."""
import random

import pytest
from PIL import Image, ImageDraw

from glmocr import deskew
from glmocr.corpus import ROOT


def page():
    """Rows of word-sized dark blocks on a white A4 page at 100 dpi."""
    rng = random.Random(0)
    im = Image.new("RGB", (827, 1169), "white"); d = ImageDraw.Draw(im)
    for y in range(90, 1080, 26):
        x = 80
        while x < 740:
            w = rng.randint(15, 70)
            d.rectangle([x, y, min(x + w, 747), y + 11], fill=(30, 30, 30))
            x += w + rng.randint(8, 14)
    return im


@pytest.mark.parametrize("deg", [-15.0, -6.0, -1.5, 0.0, 1.0, 3.0, 12.0])
def test_angle_is_found(deg):
    turned = page().rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor="white")
    assert deskew.angle(turned) == pytest.approx(deg, abs=0.15)


def test_straighten_undoes_the_turn():
    turned = page().rotate(8.0, resample=Image.BICUBIC, expand=True, fillcolor="white")
    out, a = deskew.straighten(turned, min_deg=1.0)
    assert a == pytest.approx(8.0, abs=0.15)
    assert abs(deskew.angle(out)) <= 0.15


def test_under_the_threshold_nothing_changes():
    turned = page().rotate(0.5, resample=Image.BICUBIC, expand=True, fillcolor="white")
    out, a = deskew.straighten(turned, min_deg=1.0)
    assert out is turned and abs(a) < 1.0


def test_trim_gives_back_the_canvas():
    turned = page().rotate(10.0, resample=Image.BICUBIC, expand=True, fillcolor="white")
    big, _ = deskew.straighten(turned, min_deg=1.0)
    small, _ = deskew.straighten(turned, min_deg=1.0, trim=True)
    assert small.width * small.height < 0.75 * big.width * big.height
    assert abs(deskew.angle(small)) <= 0.15


PHOTO = ROOT / "external/ocrbench_train/doc_082.png"


@pytest.mark.skipif(not PHOTO.exists(), reason="ocr-bench pages not on this machine (docs/EXTERNAL.md)")
def test_a_photo_shadow_is_not_a_line():
    """A photographed page with a dark shadow down one side: a global threshold found 2.6 degrees, the edge of the shadow;
    the owner's harness finds 5.65 by both its methods."""
    assert deskew.angle(Image.open(PHOTO)) == pytest.approx(5.65, abs=0.3)
