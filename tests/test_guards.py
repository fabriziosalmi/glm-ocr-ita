"""The rules of docs/ROADMAP.md that code can enforce."""
import json

import pytest
from PIL import Image

from glmocr import check, cli, corpus


def make_set(d, entity="1.234,56 €", text="Totale 1.234,56 €", size=(1240, 1754)):
    d.mkdir()
    Image.new("RGB", size, "white").save(d / "p0.png")
    (d / "p0.txt").write_text(text, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps([{"id": "p0", "level": 1, "lang": "it", "tpl": "t", "rot90": False,
                                              "entities": [{"type": "amount", "text": entity}]}]), encoding="utf-8")
    return str(d)


def test_every_registered_set_has_a_known_role():
    assert {s["role"] for s in corpus.registry().values()} <= {"train", "dev", "test", "spent"}
    assert corpus.role("hard") == "dev" and corpus.role("held") == "spent"


def test_a_good_set_passes(tmp_path):
    counts, bad = check.check(make_set(tmp_path / "ok"))
    assert bad == {} and counts["pages"] == 1


def test_a_broken_set_is_caught(tmp_path):
    d = make_set(tmp_path / "bad", entity="9.999,99 €", size=(800, 600))
    (tmp_path / "bad" / "stray.png").write_bytes(b"")
    _, bad = check.check(d)
    assert set(bad) == {"entity not in the ground truth", "image is not 1240x1754", "file not listed in meta.json"}


def test_escaped_line_breaks_are_caught(tmp_path):
    d = make_set(tmp_path / "esc", text="Totale\\n1.234,56 €")
    _, bad = check.check(d)
    assert list(bad) == ["ground truth holds a literal backslash-n instead of line breaks"]
    assert corpus.reference(d, "p0") == "Totale\n1.234,56 €"                      # and the loader repairs it
    mixed = make_set(tmp_path / "mixed", text="C:\\nomi\nTotale 1.234,56 €")
    assert corpus.reference(mixed, "p0") == "C:\\nomi\nTotale 1.234,56 €"         # a file with real breaks is left alone
    assert list(check.check(mixed)[1]) == ["ground truth mixes real line breaks and a literal backslash-n"]


def test_language_tied_to_level_is_caught(tmp_path):
    d = make_set(tmp_path / "tied")
    meta = [{"id": "p0", "level": 1, "lang": "it", "tpl": "t", "rot90": False, "entities": [{"type": "amount", "text": "1.234,56 €"}]},
            {"id": "p1", "level": 5, "lang": "en", "tpl": "t", "rot90": False, "entities": [{"type": "amount", "text": "1.234,56 €"}]}]
    for ext in ("png", "txt"):
        (tmp_path / "tied" / f"p1.{ext}").write_bytes((tmp_path / "tied" / f"p0.{ext}").read_bytes())
    (tmp_path / "tied" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert "a level holds one language only (language and level cannot be told apart)" in check.check(d)[1]


def test_a_test_set_cannot_be_read_before_the_ledger_says_so(tmp_path, monkeypatch):
    d = make_set(tmp_path / "frozen")
    monkeypatch.setattr(corpus, "registry", lambda: {d: {"role": "test"}})
    monkeypatch.setattr(corpus, "LEDGER", tmp_path / "LEDGER.md")
    with pytest.raises(SystemExit, match="is a test set"):   # refused before any model is loaded
        cli.main(["run", "no/such/model", "frozen_first_read", "--corpus", d])
    (tmp_path / "LEDGER.md").write_text("| 2026-10-13 | `frozen_first_read` | the question | |")
    assert corpus.ledger_has("frozen_first_read") and not corpus.ledger_has("frozen_second_look")


# ---------------------------------------------------------------- memory fences (glmocr/guard.py)
from glmocr import guard


def fake_model(d, gb):
    d.mkdir()
    with open(d / "model.safetensors", "wb") as f:
        f.truncate(int(gb * guard.GB))   # sparse: no disk used
    return str(d)


def test_only_one_model_job_at_a_time(tmp_path):
    held = guard.lock(tmp_path / "work" / ".model.lock")
    with pytest.raises(SystemExit, match="another model job is running"):
        guard.lock(tmp_path / "work" / ".model.lock")
    held.close()
    guard.lock(tmp_path / "work" / ".model.lock").close()   # free again once the first job is gone


def test_a_model_that_cannot_fit_is_refused_before_loading(tmp_path):
    m = fake_model(tmp_path / "m", 2.65)
    assert guard.preflight(m, ceiling_gb=6, available=8)[0] == pytest.approx(2.65 + guard.HEADROOM_GB)
    with pytest.raises(SystemExit, match="the ceiling is"):
        guard.preflight(m, ceiling_gb=3, available=8)
    with pytest.raises(SystemExit, match="Close something first"):
        guard.preflight(m, ceiling_gb=6, available=4.5)   # fits the ceiling, not the machine as it is now


def test_the_watch_stops_a_job_over_the_ceiling_or_a_starved_system():
    guard.watch(3.2, ceiling_gb=6, available=5)
    with pytest.raises(guard.MemoryStop, match="over the ceiling"):
        guard.watch(6.4, ceiling_gb=6, available=5)
    with pytest.raises(guard.MemoryStop, match="under the floor"):
        guard.watch(3.2, ceiling_gb=6, available=0.4)


def test_available_memory_is_read_from_the_system():
    assert 0 < guard.available_gb() < 1024


def test_linux_memory_and_a_container_limit(tmp_path):
    assert guard.linux_available("MemTotal: 16000000 kB\nMemAvailable:    8000000 kB\n") == 8000000 * 1024
    assert guard.cgroup_left(tmp_path) is None                       # no cgroup v2 files: no limit
    (tmp_path / "memory.max").write_text("max\n"); (tmp_path / "memory.current").write_text("1000\n")
    assert guard.cgroup_left(tmp_path) is None                       # a cgroup without a limit
    (tmp_path / "memory.max").write_text("6000000000\n")
    assert guard.cgroup_left(tmp_path) == 6000000000 - 1000


def test_a_gguf_is_weighed_with_its_projector(tmp_path):
    for name, gb in (("m.gguf", 0.95), ("mmproj-m.gguf", 0.87)):
        with open(tmp_path / name, "wb") as f:
            f.truncate(int(gb * guard.GB))
    assert guard.weights_gb(tmp_path / "m.gguf") == pytest.approx(1.82)


def test_resident_memory_of_a_process():
    import os
    assert 0.001 < guard.rss_gb(os.getpid()) < 64


# ---------------------------------------------------------------- NAME#N: a prefix of the sampling order
def fake_meta(n=300):
    return [{"id": f"p{i:04d}", "level": i % 5 + 1, "lang": "en" if i % 5 == 0 and i % 2 else "it" if i % 10 else "en"} for i in range(n)]


def test_sampling_order_is_fixed_nested_and_balanced():
    import collections, random
    meta = fake_meta()
    order = [m["id"] for m in corpus.sampling_order(meta)]
    shuffled = meta[:]
    random.Random(1).shuffle(shuffled)
    assert order == [m["id"] for m in corpus.sampling_order(shuffled)]      # the file order does not matter
    assert sorted(order) == sorted(m["id"] for m in meta)                   # a permutation: nothing lost, nothing twice
    first = corpus.sampling_order(meta)[:60]
    assert set(collections.Counter(m["level"] for m in first).values()) == {12}
    share = lambda pages: sum(m["lang"] == "it" for m in pages) / len(pages)
    assert abs(share(first) - share(meta)) < 0.05


def test_hash_suffix_selects_a_prefix_and_keeps_the_role():
    assert corpus.split_name("spark/itdoc_a#240") == ("spark/itdoc_a", 240)
    assert len(corpus.load_meta("held#30")) == 30
    assert {m["id"] for m in corpus.load_meta("held#30")} < {m["id"] for m in corpus.load_meta("held#60")}
    assert corpus.role("held#30") == "spent"


def test_keep_retry():
    from glmocr.engine import keep_retry
    first = {"text": "a" * 500, "looped": True}
    assert keep_retry(first, {"text": "short", "looped": False})          # a second reading that does not loop wins
    assert keep_retry(first, {"text": "b" * 900, "looped": True})        # both loop: more text before the loop
    assert not keep_retry(first, {"text": "b" * 100, "looped": True})


def test_an_installed_package_without_a_checkout(tmp_path):
    """pip or Docker: no datasets.toml beside the code. No registered sets, results/ and work/ outside site-packages."""
    import os, subprocess, sys
    env = {**os.environ, "GLMOCR_REPO": str(tmp_path / "none"), "HOME": str(tmp_path)}
    env.pop("GLMOCR_WORK", None)
    out = subprocess.run([sys.executable, "-c", "from glmocr import corpus; print(corpus.CHECKOUT, corpus.registry(), corpus.RESULTS, corpus.WORK)"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out[:2] == ["False", "{}"]
    assert out[2] == str((tmp_path / "results").resolve()) and out[3] == str(tmp_path / ".cache" / "glmocr")
