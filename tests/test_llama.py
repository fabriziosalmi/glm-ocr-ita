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


def test_wide_retry_runs_only_when_first_retry_cuts_a_loop(tmp_path, monkeypatch):
    """A page whose 1.1 retry still cuts a verbatim loop gets one wide-context retry (1.3, 256); a page whose 1.1 retry
    clears the loop, or only hit the token limit with loop_cut=None, stops after two reads."""
    from PIL import Image
    from glmocr import engine, guard
    img = tmp_path / "p.png"; Image.new("RGB", (32, 32), "white").save(img)
    monkeypatch.setattr(guard, "watch", lambda *a: None)

    for cls in (engine.Reader, llama.LlamaReader):
        r = object.__new__(cls)
        r.retry_penalty, r.ceiling, r._tmp = 1.1, 6.0, type("T", (), {"name": str(tmp_path)})()
        if cls is llama.LlamaReader:
            r.rss_gb = lambda: 1.0
        calls = []
        def fake_gen(path, penalty, task="text", context=None):
            calls.append((penalty, context))
            if penalty is None:
                return {"text": "a" * 50, "loop_cut": 318, "s": 1.0, "gen_tokens": 928, "looped": True, "peak_gb": 1.0}
            if context is None:
                return {"text": "a" * 40, "loop_cut": 318, "s": 1.0, "gen_tokens": 928, "looped": True, "peak_gb": 1.0}
            return {"text": "full form text", "loop_cut": None, "s": 2.0, "gen_tokens": 2072, "looped": False, "peak_gb": 1.0}
        r._generate = fake_gen
        out = r.read(img)
        assert calls == [(None, None), (1.1, None), (engine.WIDE_PENALTY, engine.WIDE_CTX)]
        assert out["text"] == "full form text" and out["retry"]["kept"] == "third" and out["s"] == 4.0

        # When the 1.1 retry has loop_cut=None (even if looped=True by token budget), no third read is made
        calls.clear()
        def fake_gen_nocut(path, penalty, task="text", context=None):
            calls.append((penalty, context))
            return {"text": "table", "loop_cut": None, "s": 1.0, "gen_tokens": 3000, "looped": True, "peak_gb": 1.0}
        r._generate = fake_gen_nocut
        r.read(img)
        assert calls == [(None, None), (1.1, None)]
