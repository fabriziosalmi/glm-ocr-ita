"""The region mode (glmocr/regions.py, glmocr/layout_worker.py) gives what was measured: the SDK's regions and formatting of the
pages read for docs/LEDGER.md. The worker tests need the SDK's environment (work/layout_venv) and the saved regions; skipped without."""
import json, pathlib

import pytest
from PIL import Image

from glmocr import regions

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / "work/layout_venv/bin/python"
PAGE = "newspaper_2a6b4fa088699701a6fa"   # prefix of an omnidoc_dev newspaper page: many regions, three kinds
need_sdk = pytest.mark.skipif(not PY.exists() or not (ROOT / "work/layout/omnidoc_dev").is_dir()
                              or not (ROOT / "results/odbdevR_v8d8e8op1_regions.json").exists(), reason="no SDK environment or saved regions")


def test_crop_on_the_sdk_grid():
    img = Image.new("RGB", (2000, 1000))
    assert regions.crop(img, [0, 0, 1000, 1000]).size == (2000, 1000)
    assert regions.crop(img, [250, 100, 500, 900]).size == (500, 800)


def test_no_layout_environment_says_how(monkeypatch):
    monkeypatch.delenv("GLMOCR_LAYOUT_PYTHON", raising=False)
    with pytest.raises(SystemExit, match=r'glmocr\[layout\]==0.1.5'):
        regions.layout_python()


@need_sdk
def test_worker_gives_the_measured_regions_and_text():
    page = sorted((ROOT / "work/layout/omnidoc_dev").glob(f"{PAGE}*.json"))[0]
    saved = json.load(open(page))
    run = json.load(open(ROOT / "results/odbdevR_v8d8e8op1_regions.json"))["pages"][saved["id"]]
    text = json.load(open(ROOT / "results/odbdevR_v8d8e8op1.json"))["pages"][saved["id"]]["text"]
    drop = regions.Layout(PY, keep=False)
    try:
        assert drop.regions(ROOT / "external/omnidoc_dev" / f"{saved['id']}.png")["regions"] == saved["regions"]
        assert drop.format(run["regions"]) == text
    finally:
        drop.close()
    keep = regions.Layout(PY, keep=True)
    try:
        kept = keep.regions(ROOT / "external/omnidoc_dev" / f"{saved['id']}.png")["regions"]
    finally:
        keep.close()
    labels = {r["label"] for r in kept} - {r["label"] for r in saved["regions"]}
    assert labels and labels <= {"header", "footer", "number", "footnote", "aside_text", "reference", "footer_image", "header_image"}
    assert all(r["task_type"] == ("skip" if r["label"].endswith("_image") else "text") for r in kept if r["label"] in labels)


@need_sdk
def test_worker_reports_a_bad_page_and_goes_on(tmp_path):
    lay = regions.Layout(PY, keep=False)
    try:
        with pytest.raises(regions.LayoutError):
            lay.regions(tmp_path / "missing.png")
        assert lay.format([]) == ""
    finally:
        lay.close()


@need_sdk
def test_worker_takes_the_sdk_even_with_this_repository_on_the_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))   # this repository's glmocr would come first
    lay = regions.Layout(PY, keep=False)
    try:
        assert lay.sdk == "0.1.5"
    finally:
        lay.close()
