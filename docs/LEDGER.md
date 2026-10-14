# Test-set ledger

Every read of a held-out set, as required by rule 2 of `ROADMAP.md`. A configuration chosen by looking at a test set is disqualified.

`glmocr run` refuses to read a set whose role in `datasets.toml` is `test` until the tag of the run appears in this file between backticks. So the order is: write the question, the threshold and the tag here, then run.

## `held/` (120 pages, seed 20261012) - spent, kept as a regression check

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-12 | v8d4e4 @0.7 | Does the configuration chosen on `hard/` hold on unseen pages? | 80.1% [74.1, 85.6], same profile by level as `hard/` |
| 2026-10-12 | v8d4e4 @1.0 | Is full resolution worth it for the compressed model? | +0.5 [-1.2, +2.6] vs 0.7x: no |
| 2026-10-12 | bf16 @0.7 | Does compression cost anything at equal resolution? | v8d4e4 -0.5 [-2.2, +1.4]: not shown |
| 2026-10-12 | v8d4e4 @0.85 | Scale sweep | +0.9 [-0.7, +2.8] vs 0.7x |
| 2026-10-12 | v8d4e4 @0.6 | Scale sweep | -1.1 [-2.8, +0.4] vs 0.7x |
| 2026-10-12 | bf16 @1.0 | What does the recommended configuration lose against the original at full resolution? | -3.1 [-5.5, -1.1] |
| 2026-10-12 | v16d4e4 @1.0 | Diagnostic: do skipped blocks at full resolution come from the encoder? | No: 12 skipped pages against 11, 11 in common |
| 2026-10-12 | v8d8e8 @1.0 | Diagnostic: does an 8-bit decoder remove them? | Yes: 0 skipped pages; -0.1 [-0.3, +0.1] vs bf16 @1.0. Found here, so it must be confirmed on a fresh set before release |

v8d4e4 at 0.7x had been chosen on `hard/` before `held/` existed. v8d8e8 at 1.0x was not: it was a diagnostic read that turned out to match the original, which is exactly the kind of result this ledger exists to flag. Eight reads are enough to retire `held/` as a decision set.

## Decisions taken before the data existed

| Date | Decision |
|---|---|
| 2026-10-12 | "Not worse" margin fixed at -1.5 points (a margin of -3 was considered and rejected: it equals the gap to the base model that the roadmap sets out to close) |
| 2026-10-12 | `itdoc_c` designated as the synthetic part of the frozen test set, before any commissioned dataset was generated |
| 2026-10-12 | SROIE 2019 test split (347 real scanned receipts, English) designated as a frozen test set for the compression tracks, before any model was run on it. The train split (626) stays available for tuning |
| 2026-10-12 | Preprocessing for SROIE fixed before the first read: images with a long side above 1754 px are reduced to 1754, others untouched (`generators/make_sroie.py`) |
| 2026-10-13 | Roles of the commissioned sets fixed in `datasets.toml` before any of them was read by a model (`DATASETS.md`). The designation of `itdoc_c` stands, for every renderer that produces it. `spark/itdoc_b` is added as a test set: a renderer no training page comes from |
| 2026-10-13 | `gemini/itdoc_a` and `itdoc_b` change role from `train` to `dev`: every commissioned set is validation or benchmark, none is training data. Taken while the three `gemini` sets are being regenerated; the pilot read of the old `gemini/itdoc_a` (60 pages, one configuration and part of a second) was discarded and its pages will no longer exist |
| 2026-10-13 | The line above is withdrawn the same day, before any read: `gemini/itdoc_a` and `itdoc_b` are `train` again. Evaluation is `spark` and `chatgpt` (dev: `spark/itdoc_a`, `chatgpt/itdoc_a`, `_b`, `_d`, `_e`; test: `spark/itdoc_b`, `chatgpt/itdoc_c`), fine-tuning is `gemini`. `gemini/itdoc_c` stays test, never trained on, reported apart because it shares a renderer with the training pages |
| 2026-10-13 | `gemini/itdoc_c` moves from `test` to `train`, before any model read it: a test set from the renderer of the training pages proves little. The test sets are `spark/itdoc_b` and `chatgpt/itdoc_c`. The designation of `itdoc_c` of 2026-10-12 stands for the `chatgpt` rendering |
| 2026-10-13 | Roles of the sets that arrived the same day: `gemini/itdoc_g` to `_k` are `train`; `chatgpt/itdoc_f` to `_z` are `dev`. Before this, the original model had read their first page only (and 16 pages of `gemini/itdoc_j`), as a probe |
| 2026-10-14 | `gemini/itdoc_l` to `_z` are `train`, by the rule already fixed for that renderer. They arrived overnight; no model has read any page of them. `gemini` is complete: 26 sets, 39000 pages, all passing `glmocr check` |
| 2026-10-14 | Roles of `spark/itdoc_c` to `_k`, fixed by the owner while no model had read any of their pages: `c`, `d`, `e`, `f`, `g` are `dev`; `h`, `i`, `j`, `k` are `test`. `spark/itdoc_b` stays `test`; it was read once on 2026-10-13 with mlx-vlm's prompt, so a second read needs its own entry and reason here |
| 2026-10-14 | `external/xfund_it_train` (149 real Italian forms) is `dev`, and is never trained on: its license (CC BY-NC-SA 4.0) must not reach the MIT weights. Fixed before any model read it |

## `spark/itdoc_b` and `chatgpt/itdoc_c` - open

Written on 2026-10-13 before the first read of either set. Environment: `requirements.lock`.

**Pages.** `spark/itdoc_b#120`: the first 120 pages of the sampling order (24 per level, the set's own language mix). `chatgpt/itdoc_c`: all 24 pages. If a verdict on `spark/itdoc_b` is "not shown", one extension to `#240` is allowed, for the same configurations and the same questions; the 120 added pages are unread until then. No other extension.

**Configurations and tags.** BASE `test_sparkB_bf16_1.0`, `test_chatgptC_bf16_1.0`; ACC (`v8d8e8` at full resolution) `test_sparkB_v8d8e8_1.0`, `test_chatgptC_v8d8e8_1.0`; Q4 `test_sparkB_v16d4e4_1.0`, `test_chatgptC_v16d4e4_1.0`. NOW is not read: after the pilot it is not a candidate.

**Questions and thresholds**, each answered on each set apart:

1. Is ACC not worse than BASE? Passes if the lower bound of ACC - BASE is above -1.5, with no more pages under 90% of the text than BASE + 2 per 100 and no more loops than BASE.
2. Is ACC better than Q4? Passes if the lower bound of ACC - Q4 is above 0.
3. Baseline for Track C: entity recall of BASE on the Italian and on the English pages of `spark/itdoc_b#120`. A fine-tuned model will be compared with these, read in the same environment. On `chatgpt/itdoc_c` language is tied to level, so no by-language number is taken from it.

ACC counts as confirmed if question 1 passes on both sets. It was found on `held/` and matched BASE on the `dev` pilot of the same day; nothing was tuned between the pilot and this read.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-13 | `test_sparkB_bf16_1.0`, `test_sparkB_v8d8e8_1.0` | 1, on `spark/itdoc_b#120` | **Passes.** ACC - BASE = 0.0 [-0.6, +0.6]; 88.8% [86.2, 91.2] against 88.8% [86.3, 91.3]. Pages under 90% of the text 18 against 17 of 120 (allowed: 17 + 2.4); loops 0 and 0 |
| 2026-10-13 | `test_sparkB_bf16_1.0` | 3 | BASE on `spark/itdoc_b#120`: Italian 89.7%, English 85.5% (ACC: 89.9% and 84.8%) |
| 2026-10-13 | `test_sparkB_v16d4e4_1.0` | 2, on `spark/itdoc_b#120` | **Not shown.** ACC - Q4 = +1.1 [-0.6, +2.5]; Q4 87.7% [85.3, 89.8], and Q4 - BASE = -1.1 [-2.6, +0.5]. Pages under 90% of the text: Q4 37 of 120, against 18 for ACC and 17 for BASE. The extension to `#240` has not been used |
| 2026-10-13 | `test_chatgptC_bf16_1.0`, `test_chatgptC_v8d8e8_1.0` | 1, on `chatgpt/itdoc_c` | **Passes.** ACC - BASE = +0.2 [-0.5, +1.3]; 93.1% [90.9, 95.3] against 92.9% [90.6, 95.2]. Pages under 90% of the text 0 and 0 of 24; loops 0 and 0 |
| 2026-10-13 | | 1, both sets | **ACC is confirmed**: `v8d8e8` at full resolution is not worse than the original on both test sets |
| 2026-10-13 | `test_chatgptC_v16d4e4_1.0` | 2, on `chatgpt/itdoc_c` | **Not shown.** ACC - Q4 = +1.5 [-0.2, +3.5]; Q4 91.6% [88.8, 94.4], and Q4 - BASE = -1.3 [-3.5, +0.9]. Q4 has 2 pages under 90% of the text and 1 loop; ACC and BASE have none |
| 2026-10-13 | | 2, both sets | Not shown on either set. ACC is ahead of Q4 on both (+1.1 and +1.5) with lower bounds just under zero. All reads of this round are complete |

The read of `test_sparkB_v16d4e4_1.0` stopped at page 48 with a Metal GPU timeout (`kIOGPUCommandBufferCallbackErrorTimeout`), not a memory stop. It was resumed from page 49 with the same command: decoding is deterministic, the 48 pages were not read again and no score had been looked at.

## Track C, the fine-tuned model: `spark/itdoc_b` and `chatgpt/itdoc_c` - closed, won

Written on 2026-10-14, before the validation read of the candidate had finished and before any read below. Environment: `requirements.lock`.

**Candidate.** `models/v8d8e8_op1`: adapter `work/ft/op1_90` (run `op1`, `docs/ROADMAP.md`) fused into the bf16 weights by `experiments/ft_merge.py`, then quantized as `v8d8e8`. It is read only if, on the dev pages (`spark/itdoc_a#60`, `chatgpt/itdoc_a`), it is not worse than the adapter on the bf16 model (lower bound of the difference above -1.5 on both): if quantization ate the gain, there is nothing to test. The reference is BASE with the official prompt; every read here uses the official prompt.

**Why `spark/itdoc_b` is read again.** It was read on 2026-10-13 with mlx-vlm's prompt, for the compression question. No training page comes from `spark`, and nothing in run `op1` was chosen by looking at it. BASE is read again with the official prompt so that both sides of every difference come from the same prompt.

**Pages.** `spark/itdoc_b#120` and all 24 pages of `chatgpt/itdoc_c`, as in the round above. No extension.

**Tags.** BASE `test_sparkB_bf16_1.0_op`, `test_chatgptC_bf16_1.0_op`; candidate `test_sparkB_v8d8e8op1_1.0_op`, `test_chatgptC_v8d8e8op1_1.0_op`.

**Questions and thresholds.** Differences are candidate - BASE, paired page bootstrap as `glmocr compare` (2000 resamples, seed 0); by language, the same bootstrap on the pages of that language.

1. Italian higher, on `spark/itdoc_b#120`: lower bound of the difference on the Italian pages above 0.
2. English not broken, on `spark/itdoc_b#120`: lower bound of the difference on the English pages above -1.5.
3. Not worse on `chatgpt/itdoc_c` (language tied to level, so taken whole): lower bound above -1.5.
4. On each set: no more pages under 90% of the text than BASE + 2 per 100, and no new loop on a page where the candidate finds fewer entities than BASE.

**Track C is won** if 1, 2, 3 and 4 all pass. If 1 is "not shown" with the rest passing, the claim is "not worse, Italian higher on dev only", and the test sets are not read again for this candidate.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-14 | | gate | Passed on dev before any test read: `v8d8e8_op1` against the adapter on bf16, -0.4 [-1.2, +0.1] on `spark/itdoc_a#60` and -0.4 [-0.9, +0.1] on `chatgpt/itdoc_a`; against BASE +2.5 [+0.7, +4.6] and +0.5 [-0.1, +1.1] |
| 2026-10-14 | `test_sparkB_bf16_1.0_op`, `test_sparkB_v8d8e8op1_1.0_op` | 1, on `spark/itdoc_b#120` | **Passes.** Italian (95 pages) 88.7% -> 92.4%, +3.7 [+2.4, +5.1]. All pages 88.3% -> 92.7%, +4.4 [+3.2, +5.8]; 49 pages gain entities, 3 lose one each |
| 2026-10-14 | same | 2, on `spark/itdoc_b#120` | **Passes.** English (25 pages) 86.8% -> 93.9%, +7.1 [+4.1, +10.5] |
| 2026-10-14 | same | 4, on `spark/itdoc_b#120` | **Passes.** Pages under 90% of the text 25 -> 3 of 120. Two new loops, both cut by the reader and both on pages where the candidate finds as many entities as BASE (13 of 18, 15 of 22) |
| 2026-10-14 | `test_chatgptC_bf16_1.0_op`, `test_chatgptC_v8d8e8op1_1.0_op` | 3, on `chatgpt/itdoc_c` | **Passes.** 93.0% -> 93.7%, +0.7 [+0.0, +1.4], not worse (the lower bound is at zero, not above it: not "better"). 10 pages gain entities, 5 lose (worst -2). Italian pages 94.5% -> 94.8%, English 85.7% -> 88.2%; accented words 63.8% -> 68.8% of 378 |
| 2026-10-14 | same | 4, on `chatgpt/itdoc_c` | **Passes.** Pages under 90% of the text 0 and 0; loops 0 and 0 |
| 2026-10-14 | | Track C | **Won.** Questions 1 to 4 pass. `models/v8d8e8_op1` (1184 MB, the size of `v8d8e8`) reads Italian better than the original on `spark/itdoc_b#120` and English better too, and is not worse on `chatgpt/itdoc_c`. All pages synthetic. The test sets are not read again for this candidate |

## `external/sroie_test` (347 receipts, 1388 entities) - open

Exact-string ceiling: 1331 of the 1388 entities appear verbatim in the dataset's own transcription (95.9%), so no model can score above that.

Questions and thresholds, written before the first read:

Tags: BASE `sroie_bf16_1.0`, Q4 `sroie_v16d4e4_1.0`, NOW `sroie_v8d4e4_0.7`, ACC `sroie_v8d8e8_1.0`.

1. Baseline: BASE, Q4 and NOW on real scans.
2. Is ACC (`v8d8e8` at full resolution, found on `held/`) not worse than BASE? Passes if the lower bound of ACC - BASE is above -1.5.
3. Is ACC better than Q4? Passes if the lower bound of ACC - Q4 is above 0.
4. Is NOW not worse than Q4? Passes if the lower bound of NOW - Q4 is above -1.5.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-12 | `sroie_v8d4e4_0.7` (NOW) | Questions 1 and 4, NOW side | 90.3% entity recall on 347 receipts. No comparison yet: the other three reads are not done |

Logged on 2026-10-13, a day late: the number was written in `STATUS.md` and not here. The raw output of that run was in `results/sroie_*`, which is not in git, and is not on this machine after the repository was cloned again; the figure is the one recorded at the time and cannot be recomputed until the run is repeated. Repeating it is the same read, not a second one: decoding is deterministic.

A partial run of ACC (201 of 347 receipts) was lost the same way.

### The fine-tuned model on real receipts, a subset - closed

Written on 2026-10-14, before any read of these receipts by either model. The whole split is about three hours per model, so the owner asked for a subset.

**Pages.** `external/sroie_test#100`: the first 100 receipts of the fixed sampling order (`corpus.sampling_order`, seed 20261013; every receipt is level 1 and English, so it is a seeded shuffle by id). No extension.

**Where.** On the second machine (Mac mini M4, 16 GB), same `requirements.lock`, Python 3.12.13 against 3.12.15 here. Both sides of the difference are read there, so the pairing holds even if outputs differ between machines. Archive checked against the SHA-256 above; the conversion gives 347 receipts and 1388 entities, as on 2026-10-12.

**Tags.** BASE `sroie100_bf16_1.0_op`; candidate `sroie100_v8d8e8op1_1.0_op` (`models/v8d8e8_op1`). Official prompt.

**Question.** Is the candidate not worse than BASE on real scanned receipts in English? Passes if the lower bound of candidate - BASE is above -1.5, with no more loops than BASE + 2 and no new loop on a receipt where the candidate finds fewer entities. The training never saw a receipt: this is the check that the Italian fine-tuning did not cost anything on real scans.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-14 | `sroie100_bf16_1.0_op`, `sroie100_v8d8e8op1_1.0_op` | not worse on real receipts | **Passes.** 92.0% and 92.0% of the entities, +0.0 [-1.2, +1.5]; 3 receipts lose one entity, 2 gain one. Receipts under 90% of the text 2 -> 0, loops 0 and 0. By field: address 79 -> 80%, company 94 -> 91%, date 96 -> 97%, total 99 -> 100%. 8.6 s per receipt against 9.4, peak 2.2 GB against 3.2 (Mac mini) |

## `external/xfund_it_val` (50 real Italian forms, XFUND) - closed

Written on 2026-10-14, before any model read any of these forms. The first real Italian documents of this work (`docs/EXTERNAL.md`). Read on the Mac mini, both sides there, official prompt. All 50 forms, no subset, no extension.

**Tags.** BASE `xfund_it_bf16_1.0_op` (`work/glm-ocr-bf16`); release `xfund_it_v8d8e8op1_1.0_op` (`models/v8d8e8_op1`, v1.0.0).

**Entities.** Every annotated segment of at least 2 characters once whitespace is removed, typed by the dataset's label: 1512 `answer` (the values filled in), 1222 `question` (printed field names), 65 `header`, 1115 `other`. A segment counts as found if its exact text, whitespace removed, is in the output. Segments that hold check-box marks (`O`, `x`, a square) can be missed by both models for that reason alone; the comparison is paired, so it is fair, but the absolute recall is a floor.

**Questions and thresholds**, differences release - BASE, paired bootstrap over forms (`experiments/compare_lang.py --by type` for 1; `glmocr compare` for 2):

1. Filled-in values (`answer`): is the release not worse than BASE? Passes if the lower bound is above -1.5. If it is above 0, the claim is "better".
2. All entities: the same threshold.
3. No more forms under 90% of the text than BASE + 1, and no new loop on a form where the release finds fewer entities.

Whatever the outcome, it goes in the README and in the model card, next to the synthetic results.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
| 2026-10-14 | `xfund_it_bf16_1.0_op`, `xfund_it_v8d8e8op1_1.0_op` | 1, filled-in values | **Not shown.** 76.9% -> 74.5%, -2.4 [-8.3, +2.1] on 1512 values |
| 2026-10-14 | same | 2, all entities | **Not shown.** 74.6% -> 74.9%, +0.3 [-3.0, +2.9] on 3914; 29 forms gain entities, 5 lose. By type: printed field names 86.7 -> 86.6%, headers 75.4 -> 78.5%, other text 58.0 -> 62.4% (+4.4 [+1.9, +7.2]) |
| 2026-10-14 | same | 3 | **Fails.** Forms under 90% of the text 12 -> 8 (passes), but two new loops on forms where the release finds fewer entities: `it_val_29` (74 -> 50 of 121) and `it_val_32` (105 -> 55 of 168). Both start on a comb field (digits in boxes, written `3\|4\|3\|4...`), the reader cuts them, and the rest of the form is lost. Loops 2 -> 3 overall; one of BASE's two is gone |
| 2026-10-14 | | verdict | **The release is not shown better or worse than the original on real Italian forms**, and it loops on comb fields where the original does not, twice in 50. Said as such in the README and the model card. Two forms carry the loss: over the other 48 the release finds 87 entities more, but that split was made after seeing the result and is not a claim. Next: study loops on comb fields on the XFUND Italian training split (149 forms), never on this one |

## `external/omnidoc_200` (OmniDocBench v1.5, 200 pages) - open

Written on 2026-10-14, before any model read any page of the benchmark; the evaluator was checked first with the annotation's own text as prediction. Read on the Mac mini, both sides there, official prompt, full resolution, pages as they are.

**Why.** The fine-tuning taught Italian and English documents. This checks that it did not cost the model what it was built for: Chinese and English pages of every kind, scored by the benchmark's own code.

**Tags.** BASE `omnidoc200_bf16_1.0_op`; release `omnidoc200_v8d8e8op1_1.0_op`.

**Measure.** Normalized edit distance of the text blocks, official evaluator (`experiments/omnidoc_eval.py`), per page = mean over its matched blocks; lower is better. Difference = BASE - release, times 100, so that positive means the release is better; paired bootstrap over pages (2000, seed 0).

**Questions and thresholds**, on the English pages (150) and on the Chinese pages (50) apart:

1. Is the release not worse than BASE? Passes if the lower bound of the difference is above -1.5.
2. Reading order edit distance: reported, no threshold.

| Date | Configuration | Question | Outcome |
|---|---|---|---|
