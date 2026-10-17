"""glmocr/llama.py without a server: which reader a model path selects, and the prompt it writes."""
import platform
import sys

import pytest

from glmocr import cli, llama
from glmocr.engine import Reader


def test_a_gguf_path_selects_llama_cpp():
    assert cli.reader_class("work/gguf/glm-ocr-ita-Q8_0.gguf") is llama.LlamaReader
    if (sys.platform, platform.machine()) == ("darwin", "arm64"):
        assert cli.reader_class("models/v8d8e8_op1") is Reader
    else:   # Linux, Windows, an Intel Mac: no MLX, a clear refusal instead of an ImportError
        with pytest.raises(SystemExit, match="Apple Silicon only"):
            cli.reader_class("models/v8d8e8_op1")


def test_the_prompt_is_the_official_one_with_its_prefix():
    """llama-server adds no [gMASK]<sop>; without them the page loops (2026-10-17). mtmd puts the image markers around the media."""
    p = llama.PROMPT.format(media="<M>", task="Text Recognition:")
    assert p == "[gMASK]<sop><|user|>\n<M>Text Recognition:<|assistant|>\n"


def test_one_projector_next_to_the_model(tmp_path):
    m = tmp_path / "model.gguf"; m.touch()
    with pytest.raises(SystemExit):
        llama.mmproj_for(m)
    (tmp_path / "mmproj-a.gguf").touch()
    assert llama.mmproj_for(m).name == "mmproj-a.gguf"
    (tmp_path / "mmproj-b.gguf").touch()
    with pytest.raises(SystemExit):
        llama.mmproj_for(m)


def test_the_gguf_release_is_recognized_and_is_the_default_off_apple_silicon(monkeypatch):
    assert llama.is_gguf(llama.RELEASE_GGUF) and llama.is_gguf("x/y.gguf")
    assert not llama.is_gguf("fabriziosalmi/GLM-OCR-ita-mlx-8bit") and not llama.is_gguf("models/v8d8e8_op1")
    assert cli.reader_class(llama.RELEASE_GGUF) is llama.LlamaReader
    monkeypatch.setattr(cli, "apple_silicon", lambda: False)
    assert cli.default_model() == llama.RELEASE_GGUF
    monkeypatch.setattr(cli, "apple_silicon", lambda: True)
    assert cli.default_model() == cli.RELEASE


def test_a_local_gguf_is_not_downloaded(tmp_path):
    m = tmp_path / "m.gguf"
    assert llama.fetch(m) == m
