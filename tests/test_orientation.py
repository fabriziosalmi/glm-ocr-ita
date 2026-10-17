"""A phone photo is stored turned, with an EXIF orientation: every reader must see it upright."""
from PIL import Image

from glmocr.engine import open_page, turned_on_disk


def photo(tmp_path, orientation):
    im = Image.new("RGB", (60, 30), "white")   # stored landscape, as the camera wrote it
    exif = Image.Exif(); exif[0x0112] = orientation
    path = tmp_path / f"o{orientation}.png"; im.save(path, exif=exif)
    return path


def test_a_turned_photo_is_opened_upright(tmp_path):
    p = photo(tmp_path, 6)
    assert turned_on_disk(p)
    assert open_page(p).size == (30, 60)
    assert open_page(Image.open(p)).size == (30, 60)


def test_an_upright_file_is_left_as_it_is(tmp_path):
    p = photo(tmp_path, 1)
    assert not turned_on_disk(p)
    assert open_page(p).size == (60, 30)
