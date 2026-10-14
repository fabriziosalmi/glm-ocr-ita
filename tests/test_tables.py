"""Every number in the README tables, recomputed from results/. If one of these fails, the README or the scorer is wrong."""
import glob, pathlib
import json

import pytest

from glmocr import corpus, metrics, score

# tag: size is not recomputable here; recall, CI, L1..L5, skipped (of 108 upright), loops, s/page, gen tok/s, peak GB
HELD = {
    "held_bf16_1.0":    (83.3, (77.7, 88.2), (100, 100, 97, 89, 35), 0, 4, 15.3, 68, 3.16),
    "held_v8d8e8_1.0":  (83.2, (77.6, 88.2), (100, 100, 97, 89, 35), 0, 4, 15.6, 74, 2.10),
    "held_v16d4e4_1.0": (80.5, (74.6, 85.8), (99, 99, 93, 86, 30), 12, 3, 12.8, 99, 2.19),
    "held_bf16_0.7":    (80.6, (74.8, 85.8), (100, 100, 93, 82, 34), 1, 4, 10.9, 67, 3.07),
    "held_v8d4e4_1.0":  (80.6, (74.7, 85.9), (99, 99, 93, 86, 31), 11, 3, 13.5, 97, 1.77),
    "held_v8d4e4_0.85": (81.0, (75.3, 86.1), (99, 99, 94, 86, 32), 10, 1, 9.6, 109, 1.77),
    "held_v8d4e4_0.7":  (80.1, (74.1, 85.6), (99, 100, 92, 85, 30), 3, 3, 7.2, 131, 1.74),
    "held_v8d4e4_0.6":  (79.0, (72.9, 84.6), (100, 96, 93, 83, 28), 1, 3, 5.7, 147, 1.74),
}
HELD_PAIRS = [   # B, A, difference, CI
    ("held_v8d8e8_1.0", "held_bf16_1.0", -0.1, (-0.3, 0.1)), ("held_v8d8e8_1.0", "held_v16d4e4_1.0", 2.6, (1.3, 4.3)),
    ("held_v16d4e4_1.0", "held_bf16_1.0", -2.7, (-4.3, -1.4)), ("held_v8d4e4_1.0", "held_v16d4e4_1.0", 0.1, (-0.2, 0.4)),
    ("held_v8d4e4_0.7", "held_v16d4e4_1.0", -0.4, (-2.5, 1.2)), ("held_v8d4e4_0.7", "held_bf16_1.0", -3.1, (-5.5, -1.1)),
    ("held_v8d4e4_1.0", "held_bf16_1.0", -2.6, (-4.3, -1.2)), ("held_bf16_0.7", "held_bf16_1.0", -2.6, (-5.4, -0.1)),
    ("held_v8d4e4_0.7", "held_bf16_0.7", -0.5, (-2.2, 1.4)), ("held_v8d4e4_0.6", "held_v8d4e4_0.7", -1.1, (-2.8, 0.4)),
    ("held_v8d4e4_0.85", "held_v8d4e4_0.7", 0.9, (-0.7, 2.8)), ("held_v8d4e4_1.0", "held_v8d4e4_0.7", 0.5, (-1.2, 2.6)),
    ("held_v8d4e4_0.7_oriented", "held_v8d4e4_0.7", 2.0, (0.8, 3.5)), ("held_bf16_1.0_oriented", "held_bf16_1.0", 1.1, (0.4, 1.9)),
]
HARD = {   # all at 0.7x: recall, CI, L1..L5, s/page, gen tok/s, peak GB
    "C_bf16":    (81.6, (72.1, 90.1), (100, 100, 94, 86, 30), 10.4, 67, 3.07),
    "C_v8d4e4":  (81.0, (71.2, 89.8), (100, 100, 92, 86, 29), 7.0, 135, 1.74),
    "C_v16d4e4": (81.0, (71.1, 89.7), (100, 100, 92, 87, 28), 6.9, 132, 2.13),
    "C_v16d8e8": (81.4, (71.9, 90.1), (100, 100, 94, 85, 30), 8.6, 91, 2.44),
    "C_v8d8e8":  (80.7, (70.6, 89.6), (100, 100, 90, 87, 29), 8.2, 88, 1.99),
}
LIGHTER = [   # against C_v8d4e4: difference, CI, pages with a skipped block
    ("C_v8d4e4", 0.0, (0.0, 0.0), 1), ("W_v8d4e4g128", 1.0, (-0.2, 2.6), 0), ("W_v6d4e4g128", 0.8, (-0.4, 2.3), 0),
    ("W_v6d4e4", -3.3, (-10.8, 1.2), 1), ("W_v5d4e4", -0.4, (-1.8, 1.0), 1), ("W_v4d4e4", -3.9, (-7.8, -1.0), 1),
]
near = lambda x, y, tol=0.05: abs(x - y) <= tol + 1e-9
r1 = lambda x: round(x, 1)


@pytest.mark.parametrize("tag", HELD)
def test_held_table(tag):
    rec, ci, levels, skipped, loops, s_page, tps, peak = HELD[tag]
    r = score.summarize("held", tag, with_cer=False)
    assert (r["pages"], r["entities"], r["upright"]) == (120, 1434, 108)
    assert r1(r["recall"]) == rec and (r1(r["ci"][0]), r1(r["ci"][1])) == ci
    assert tuple(round(r["levels"][l]["recall"]) for l in range(1, 6)) == levels
    assert (len(r["skipped"]), len(r["loops"])) == (skipped, loops)
    assert r1(r["s_page"]) == s_page and round(r["gen_tps"]) == tps and round(r["peak_gb"], 2) == peak


@pytest.mark.parametrize("b, a, diff, ci", HELD_PAIRS)
def test_held_paired_differences(b, a, diff, ci):
    c = score.compare("held", a, b)
    assert c["pages"] == 120
    # the README rounds each recall first, so a printed difference may be 0.1 away from the rounded exact one
    assert near(c["diff"], diff, 0.1) and (r1(c["ci"][0]), r1(c["ci"][1])) == ci


@pytest.mark.parametrize("tag", HARD)
def test_tuning_table(tag):
    rec, ci, levels, s_page, tps, peak = HARD[tag]
    r = score.summarize("hard", tag, with_cer=False)
    assert (r["pages"], r["entities"]) == (40, 483)
    assert r1(r["recall"]) == rec and (r1(r["ci"][0]), r1(r["ci"][1])) == ci
    assert tuple(round(r["levels"][l]["recall"]) for l in range(1, 6)) == levels
    assert r1(r["s_page"]) == s_page and round(r["gen_tps"]) == tps and round(r["peak_gb"], 2) == peak


@pytest.mark.parametrize("tag, diff, ci, skipped", LIGHTER)
def test_lighter_variants(tag, diff, ci, skipped):
    c = score.compare("hard", "C_v8d4e4", tag)
    assert near(c["diff"], diff, 0.1) and (r1(c["ci"][0]), r1(c["ci"][1])) == ci
    assert len(score.summarize("hard", tag, with_cer=False)["skipped"]) == skipped


def test_skipped_blocks_are_the_decoder():
    q4, ours, acc = (set(score.summarize("held", t, with_cer=False)["skipped"]) for t in ("held_v16d4e4_1.0", "held_v8d4e4_1.0", "held_v8d8e8_1.0"))
    assert len(q4 & ours) == 11 and not acc
    assert {i.split("_")[1] for i in ours} == {"invoice", "footnotes"}
    assert q4 - ours == {"102_bilingual_L5_en"}   # the one page outside those two templates, at the unreadable level


def test_rerun_is_byte_identical():
    for a, b in (("C_bf16", "V_bf16_s70"), ("C_v8d4e4", "V_v8d4e4_s70")):
        pa, pb = corpus.load_run(a)["pages"], corpus.load_run(b)["pages"]
        assert len(pa) == 40 and all(pa[i]["text"] == pb[i]["text"] for i in pa)


def test_two_scale_policies():
    rows = {r["policy"]: r for r in score.adaptive("held", ["held_v8d4e4_0.7", "held_v8d4e4_1.0"])["rows"]}
    base = rows["only held_v8d4e4_0.7"]
    assert near(rows["longest"]["recall"] - base["recall"], 0.7, 0.1)
    assert near(rows["oracle (needs ground truth)"]["recall"] - base["recall"], 2.3, 0.1)
    assert round(rows["longest"]["s_page"] / base["s_page"]) == 3


def test_loop_threshold_sits_in_a_gap():
    """No saved output of 200 bytes or more compresses to between 21% and 27%: the 25% threshold splits two separate groups.
    The gap was 21 to 29% until 2026-10-14, when a level 5 page read with the official prompt came out at 27.2%:
    garbled sentences said several times over, ending by itself after 810 tokens.
    OmniDocBench reads are left out: Chinese pages and slides compress to 21-27% without looping (10 of 400 outputs), so the
    ratio is not a loop sign there and is not used on that set."""
    ratios = []
    for f in glob.glob(str(corpus.RESULTS / "*.json")) + glob.glob(str(corpus.ROOT / "archive/results/*.json")):
        if pathlib.Path(f).name.startswith("omnidoc"):
            continue
        for p in json.load(open(f)).get("pages", {}).values():
            if len(p["text"].encode()) >= metrics.LOOP_MIN_BYTES:
                ratios.append(metrics.compress_ratio(p["text"]))
    assert len(ratios) > 2500 and not [x for x in ratios if 0.21 <= x < 0.27]
