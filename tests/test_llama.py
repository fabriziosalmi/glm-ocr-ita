"""glmocr/llama.py without a server: which reader a model path selects, and the prompt it writes."""
import pytest

from glmocr import cli, llama
from glmocr.engine import Reader


def test_a_gguf_path_selects_llama_cpp():
    assert cli.reader_class("work/gguf/glm-ocr-ita-Q8_0.gguf") is llama.LlamaReader
    assert cli.reader_class("models/v8d8e8_op1") is Reader


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
