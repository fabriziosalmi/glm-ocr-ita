"""Every number of the release tables in the README (weights `v8d8e8_op1`: v1.0.0, and v1.1.0 = the same weights read with the
retry at repetition penalty 1.1), recomputed from results/. The page sets are not in this repository (docs/DATASETS.md): each test
is skipped where its set is not on the machine, and SROIE and XFUND outputs are not in git at all (docs/EXTERNAL.md)."""
import pytest

from glmocr import corpus, metrics, score

r1 = lambda x: round(x, 1)
BASE, OURS = "bf16_1.0_op", "v8d8e8op1_1.0_op"
# corpus, tag prefix: recall A, recall B, difference, CI, pages under 90% A / B, loops A / B
ROWS = {
    "spark/itdoc_b#120": ("test_sparkB_", 88.3, 92.7, 4.4, (3.2, 5.8), 25, 3, 0, 2),
    "chatgpt/itdoc_c": ("test_chatgptC_", 93.0, 93.7, 0.7, (0.0, 1.4), 0, 0, 0, 0),
    "spark/itdoc_a#60": ("sparkA_", 87.5, 90.0, 2.5, (0.7, 4.6), 13, 5, 0, 0),
    "chatgpt/itdoc_a": ("chatgptA_", 94.2, 94.7, 0.5, (-0.1, 1.1), 0, 0, 0, 0),
    "external/sroie_test#100": ("sroie100_", 92.0, 92.0, 0.0, (-1.2, 1.5), 2, 0, 0, 0),
}
# XFUND Italian forms (test, read on the Mac mini): v1.0.0, then v1.1.0
XFUND = {"xfund_it_v8d8e8op1_1.0_op": (74.6, 74.9, 0.3, (-3.0, 2.9), 12, 8, 2, 3),
         "xfund_it_v8d8e8op1_1.0_op_rp110": (74.6, 77.7, 3.1, (1.3, 5.9), 12, 5, 2, 0)}
LANGS = {"spark/itdoc_b#120": {"it": (88.7, 92.4, 3.7, (2.4, 5.1)), "en": (86.8, 93.9, 7.1, (4.1, 10.5))}}


def runs(name):
    prefix = ROWS[name][0]
    try:
        corpus.corpus_dir(name)
    except corpus.Missing:
        pytest.skip(f"{name} is not on this machine")
    if not (corpus.result_path(prefix + BASE).exists() and corpus.result_path(prefix + OURS).exists()):
        pytest.skip(f"the runs on {name} are not in results/")
    return prefix + BASE, prefix + OURS


@pytest.mark.parametrize("name", ROWS)
def test_release_row(name):
    a, b = runs(name)
    _, ra, rb, diff, ci, skip_a, skip_b, loop_a, loop_b = ROWS[name]
    c = score.compare(name, a, b)
    assert (r1(c["recall_a"]), r1(c["recall_b"]), r1(c["diff"])) == (ra, rb, diff)
    assert (r1(c["ci"][0]), r1(c["ci"][1])) == ci
    sa, sb = score.summarize(name, a, with_cer=False), score.summarize(name, b, with_cer=False)
    assert (len(sa["skipped"]), len(sb["skipped"]), len(sa["loops"]), len(sb["loops"])) == (skip_a, skip_b, loop_a, loop_b)


def test_new_loops_cost_no_entity():
    """The two loops of the release model on the spark test set are cut by the reader, on pages where it finds as many entities as the original."""
    name = "spark/itdoc_b#120"
    a, b = runs(name)
    pa, pb = corpus.load_run(a)["pages"], corpus.load_run(b)["pages"]
    loop = lambda p: p.get("loop_cut") is not None or metrics.is_loop(p["text"], p["gen_tokens"])
    for m in corpus.load_meta(name):
        if loop(pb[m["id"]]):
            assert not loop(pa[m["id"]]) and pb[m["id"]]["loop_cut"] is not None
            assert metrics.found(m["entities"], pb[m["id"]]["text"]) >= metrics.found(m["entities"], pa[m["id"]]["text"])


@pytest.mark.parametrize("name", LANGS)
def test_release_by_language(name):
    a, b = runs(name)
    meta = corpus.load_meta(name)
    ca, cb = score._counts(meta, corpus.load_run(a)["pages"]), score._counts(meta, corpus.load_run(b)["pages"])
    for lang, (ra, rb, diff, ci) in LANGS[name].items():
        ids = [m["id"] for m in meta if m["lang"] == lang]
        d = lambda s: metrics.recall(cb, s) - metrics.recall(ca, s)
        lo, hi = metrics.bootstrap(d, ids)
        assert (r1(metrics.recall(ca, ids)), r1(metrics.recall(cb, ids)), r1(d(ids)), (r1(lo), r1(hi))) == (ra, rb, diff, ci)


@pytest.mark.parametrize("tag", XFUND)
def test_xfund(tag):
    name, a = "external/xfund_it_val", "xfund_it_bf16_1.0_op"
    try:
        corpus.corpus_dir(name)
    except corpus.Missing:
        pytest.skip(f"{name} is not on this machine")
    if not (corpus.result_path(a).exists() and corpus.result_path(tag).exists()):
        pytest.skip(f"the runs on {name} are not in results/")
    ra, rb, diff, ci, skip_a, skip_b, loop_a, loop_b = XFUND[tag]
    c = score.compare(name, a, tag)
    assert (r1(c["recall_a"]), r1(c["recall_b"]), r1(c["diff"]), (r1(c["ci"][0]), r1(c["ci"][1]))) == (ra, rb, diff, ci)
    sa, sb = score.summarize(name, a, with_cer=False), score.summarize(name, tag, with_cer=False)
    assert (len(sa["skipped"]), len(sb["skipped"]), len(sa["loops"]), len(sb["loops"])) == (skip_a, skip_b, loop_a, loop_b)
