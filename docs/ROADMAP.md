# Roadmap

Written 2026-10-12. Numbers come from `results/` and are reproduced in `docs/RESULTS.md` (in the README until 2026-10-15). File names updated on 2026-10-13 for the `glmocr` package; no goal, rule, threshold or target was changed.

## Goals

1. **Compression.** Either (A) be lighter than a standard 4-bit build of GLM-OCR at the same quality, or (B) at 4 bits be faster and/or more accurate than the base model.
2. **Italian.** Read Italian documents better than the base model.

Goal 1 can be reached without training. Goal 2 cannot: quantizing and resizing can only lose information, so beating the base model needs new weights. The roadmap is ordered accordingly.

## Priorities, set by the owner on 2026-10-13

1. **Quality**, 2. **speed**, 3. **size**. And the project has won when, after a fine-tune, Italian is read better than by BASE and English is not broken: that is Track C, with its thresholds unchanged.

What follows from it, without touching any rule, threshold or target below:

- The starting point is **ACC** (`v8d8e8` at full resolution), the only compressed configuration with no known defect. It matched BASE on two new sets on 2026-10-13 (`docs/RESULTS.md`, "Pilot on the commissioned sets"): 0.0 [-0.3, +0.3] and -0.1 [-0.4, +0.3].
- **NOW is not a release candidate.** A fixed 0.7x scale loses 34 points on dense small print. Speed has to be found at full quality.
- Order of work: Phase 0 (the yardstick), then Phase 4 (Italian, training on the `gemini` sets from BASE, quantized to 8 bits afterwards), then Phase 3 (speed), then Phases 1 and 2 (size). Phase 4's own precondition, "starts only when Phases 0 to 2 are closed", is replaced by: starts when Phase 0 is closed and ACC is confirmed on `test`.
- Not pursued for now, by decision of the same day: a learned upscaler, and reading a page in zones at different scales.
- Set aside until there is a final model, by decision of the same day: the extension of the test read to 240 pages (it only serves the comparison with Q4, which can be measured later); decoder widths between 4 and 8 bits; SROIE, which becomes a receipts benchmark for the finished model. Reading a page in pieces is a line of work of its own: it gets its own branch, later.

### The first fine-tuning: plan and stopping rule

Written on 2026-10-13, before the first training step.

- **Data.** 120 pages, 20 from each of the six `gemini` sets `a` to `f`, levels 1 to 3 (`experiments/ft_data.py --per-set 20 --out external/ft_tr_120`). The first 10 pages of every set's sampling order are held back and never trained on. First planned at 240; halved the same evening, before any training, to shorten the teacher read: 200 steps then see each page less than twice.
- **Training.** Low-rank adapters (rank 8) on the decoder of BASE in bf16, vision encoder frozen, one page per step at full resolution, 200 steps, learning rate 2e-5 (`experiments/ft_train.py`). An adapter is saved every 50 steps; the ones at 100 and 200 are read (the owner chose two points instead of four, to shorten the evening). A rehearsal of 3 steps took 6.4 s per step and peaked at 7.1 GB.
- **Following it.** Each saved adapter is read on the quick panel (`experiments/panel.py`): 25 pages from `spark` and `chatgpt` ("outside"), 30 held-back `gemini` pages ("inside"), and an English guard of 14 pages (the 10 English pages of `spark/itdoc_a#60` plus the 4 English pages of "outside"). Each is compared with BASE on the same pages. No test set is in the panel.
- **Stopping rule.** Going through the adapters in order, the first one that does any of the following is discarded together with every later one:
  1. English recall on the guard more than 1.5 points below BASE;
  2. more repetition loops than BASE on "outside" plus the guard;
  3. more pages under 90% of the text than BASE on "outside".
- **What the panel can and cannot say.** It can say that something broke. It cannot declare a gain: with 25 pages a difference of 2 or 3 points is inside the noise. A surviving adapter becomes a candidate only if Italian recall on "outside" is not below BASE. The candidate is then read on the two `dev` sets the original has already been read on in this environment, `spark/itdoc_a#60` and `chatgpt/itdoc_a` (24 pages), about 36 minutes; then one read of `test` with its question written first in `LEDGER.md`.
- **Outcome of the first run (2026-10-13, `step1`).** Trained to step 190 of 200, then stopped by hand: the machine was swapping (another training job, from another session, was on the GPU at the same time). The adapters saved at 50, 100 and 150 steps are kept. Loss fell from 1.20 (steps 1 to 10) to 0.09 (11 to 20) and 0.004 by step 100. The adapter at 100 steps on the quick panel, against BASE on the same pages:

  | Group | Pages | BASE | Adapter | Difference [95% CI] | Loops | Pages under 90% |
  |---|---|---|---|---|---|---|
  | outside | 25 | 93.6% | 91.5% | -2.2 [-5.7, -0.1] | 0 -> 1 | 1 -> 1 |
  | inside | 30 | 85.0% | 84.6% | -0.4 [-1.7, +1.0] | 0 -> 0 | 0 -> 0 |
  | English guard | 14 | 86.6% | 82.9% | -3.7 [-12.8, +3.2] | 0 -> 1 | 5 -> 3 |

  **The stopping rule discards it** (English -3.7 on the guard, one new loop), and with it the adapter at 150 steps, which was not read. Page by page: 8 better, 40 equal, 17 worse. Three things happened at once. On `spark` two pages gained 3 and 4 entities with about 100 more tokens of output each: the skipped footer came back, which is what the training was for. On the dense `chatgpt` pages single characters inside codes got doubled (`...5169 9661...` read as `...5169 99661...`, `OGPPJP71...` as `OGPPJPJ71...`). And one English level 5 page that BASE read to the end became a repetition loop, which alone is 25 of the lost entities. A negative result: this recipe does not beat BASE, and it is not pursued as it is.
- **A defect in the training format, found the same evening.** The chat template renders an assistant turn as `<|assistant|>\n<think></think>\n` followed by the text, with nothing after it. When the model reads a page its answer starts straight after `<|assistant|>\n` (checked on six pages: the first tokens are page text) and ends with the token `<|user|>` (six of six). So `step1` was taught first of all to write `<think></think>`, which is what the loss falling from 1.2 to 0.09 in twenty steps was, and was never shown where to stop. Its negative result stands for that recipe with that defect, not for fine-tuning as such. `experiments/ft_train.py` now builds an answer as it looks when the model reads; `--template-format` reproduces the old behaviour.
- **Second run (`step2`), same defect, not evaluated.** 63 pages chosen because BASE misses an entity on them or returns under 90% of their words (`ft_targets.py --errors-only`, at most 15 per set; 35 Italian, 28 English), one pass, learning rate 5e-6, adapters at 21, 42 and 63 steps. Loss 1.84, 0.73, 0.22, 0.11, 0.07, 0.05. No swap, 20 minutes. Its panel read was stopped at 10 pages by the owner's decision once the defect was known: the next run changes the format only, on the same pages and settings.
- **The cause of all three failures, found the same night: in training the model could see the future.** mlx-vlm 0.7.6 calls the model with the batch's attention mask (all ones for a single page). GLM-OCR's decoder builds its causal mask only when it is given none (`models/glm_ocr/language.py`: `if mask is None: mask = create_attention_mask(...)`), so with the trainer's call every token attends to the tokens after it. Measured on the original model, on three training pages, answer = the model's own output: mean loss 1.6, 3.1 and 1.9, first choice equal to the answer on 70%, 54% and 71% of the tokens with the trainer's call; 0.008, 0.001 and 0.040, and 99.9%, 100% and 99.0% with no mask. An adapter trained that way learns to copy the next token from the future, which explains a loss near zero after a few dozen steps on pages seen once, and reads no better when the future is hidden again. Inputs were ruled out first (same prompt tokens, pixel values and grid as when reading), then reading order (0.2% of the words). With the causal mask the loss on the real training targets is 0.06, 0.81 and 0.18: that is the signal, and on the page where the model skips letterhead and footer 14% of the tokens carry it. `experiments/ft_train.py` now uses its own loss (`causal_loss`); `--trainer-mask` reproduces the old call. Runs `step1`, `step2` and `step3` are void. `step3` (corrected format, old loss; curve 1.91, 0.65, 0.25, 0.11, 0.07, 0.05) was not evaluated.
- **References found.** mlx-vlm issue #830 (GLM-OCR LoRA training broken by image tokens and position indexing, closed as fixed in May 2026) and #1021 with PR #1214 (repetition loops of GLM-OCR in mlx-vlm that the same weights did not show in Ollama, fixed in the position indexing): both say the library's GLM-OCR paths for training and for reading have diverged before. The causal-mask defect is not in either. LLaMA-Factory's `glm_ocr` template confirms the answer format (straight after `<|assistant|>`, stop on `<|user|>`). Reported upstream on 2026-10-13: https://github.com/Blaizzy/mlx-vlm/issues/2509 (still present in 0.7.8). To do: check whether loops and near-empty pages of the original model also come from the library, by reading the same pages with the reference implementation.
- **Fourth run (`step4`), the first valid one.** Same 63 pages, answers formatted as when reading, causal loss, learning rate 2e-5, one pass, 21 minutes, no swap. Loss by ten steps: 0.126, 0.049, 0.092, 0.026, 0.042, 0.034. The adapter at 21 steps on the quick panel, against BASE on the same pages:

  | Group | Pages | BASE | Adapter | Difference [95% CI] | Loops | Pages under 90% |
  |---|---|---|---|---|---|---|
  | outside | 25 | 93.6% | 94.2% | +0.5 [-0.3, +1.4] | 0 -> 0 | 1 -> 0 |
  | inside | 30 | 85.0% | 87.6% | +2.6 [-0.2, +5.6] | 0 -> 0 | 0 -> 0 |
  | English guard | 14 | 86.6% | 91.0% | +4.4 [+1.9, +8.9] | 0 -> 1 | 5 -> 0 |

  Page by page: 25 better, 33 equal, 7 worse. On the six `spark` pages where BASE returned 75 to 84% of the text the adapter returns 91 to 100%, with 4 or 5 more entities each: letterheads and footers come back. **The stopping rule discards it all the same**: one English level 5 page (`spark/itdoc_a`, `itdoc_a_1479_contract_L5_en`) that BASE closes normally now ends in a run of zeros until the token budget; it loses no entity, and it is a loop. The adapter at 63 steps was not read. So: the first sign that the training does what it was meant to, and not a candidate under the rule as written.
- **Amendment to the stopping rule, 2026-10-14, by the owner's decision, written before run `step5`.** Reason: the reader now stops a repetition as it starts (`glmocr/engine.py`, `LoopStop`; `metrics.loop_start`), so a loop no longer costs the token budget and what was read before it is kept. Clause 2 becomes: *a new loop, cut or not, on a page where the adapter finds fewer entities than BASE*. A new loop that costs no entity is reported and does not stop. Clauses 1 and 3 are unchanged. A cut loop is still counted as a loop everywhere. `step4` at 21 steps would pass the amended rule (its one new loop costs no entity); that is a judgement after the fact and it is not taken as a pass: `step5` is the run the amended rule is for.
- **Fifth run (`step5`), not started.** Planned as `step4` with half the learning rate; suspended when the prompt difference below was found, because it would have been trained and judged on the wrong prompt.
- **The prompt was not the official one, found on 2026-10-14.** Eight pages were read again with the reference implementation (transformers 5.19 and PyTorch, same weights, `work/ref_read.py`): the clean `gemini/itdoc_j` page that mlx-vlm returned almost empty (2 entities of 14) came back whole (14 of 14), and a `chatgpt/itdoc_r` page went from 68 to 78 entities of 80. The cause is not the generation code. `mlx_vlm.prompt_utils.apply_chat_template` renders the template with thinking switched off, which for GLM-OCR turns `...Text Recognition:<|assistant|>\n` into `...Text Recognition:/nothink<|assistant|>\n<think></think>\n`: four more tokens (2788 against 2784), same image tokens, same pixel values (largest difference 2e-7). With the official prompt mlx-vlm returns the same text as the reference implementation on 7 of the 8 pages and differs by one entity on the eighth. What did not change: the `chatgpt/itdoc_k` page loops with both prompts, and the skipped letterhead and footer on `spark` are identical with both, so that is the model's own behaviour. **Every number in the README of that date was measured with mlx-vlm's prompt.** The reader now uses the official one by default (`--prompt mlx` reproduces the old runs; each result file records which); result tags ending in `_op` are reads with the official prompt. The training targets and the choice of the 63 error pages were made from old-prompt reads, so run `step5` was not started: the baselines are being read again first.
- **How much the prompt matters overall, measured the same night.** The original model read again with the official prompt, paired with the old reads: `spark/itdoc_a#60` 87.5% against 88.3%, -0.8 [-1.7, -0.1]; `chatgpt/itdoc_a` 94.2% against 94.4%, -0.3 [-0.6, 0.0]; `hard/` at 0.7x 82.0% against 82.2%, -0.2 [-0.6, 0.0]. The official prompt repairs some catastrophic pages and is, on the whole, equal or slightly worse. So those tables are within a point of what the official prompt gives; they are not void. The official prompt stays the default from now on, for fidelity to the reference implementation and to how the model is trained elsewhere, and new reads are compared with `*_op` baselines only.
- **Run `op1`, planned on 2026-10-14 before any of its reads: the first fine-tuning with everything known so far put right.** Official prompt for the teacher read, for training and for the panel; answers formatted as when reading; causal loss; loops cut by the reader; the stopping rule as amended above.
  - Pages: 8 from each of `gemini/itdoc_a` to `_s` (152; levels 1 to 3; the first 10 of every set's sampling order held back), read by the original model with the official prompt (`fttr_op_bf16_1.0`). Sets `t` to `z` are left out: the audit found them near-copies of themselves. Training keeps only the pages where the original misses an entity or returns under 90% of the words, at most 6 per set.
  - Training: rank 8 on the decoder, learning rate 1e-5, one pass, adapters saved at each third of it; the first and the last are read.
  - Judged on the quick panel against the original read with the official prompt (`bf16_1.0_op`): 25 pages outside, 30 inside, English guard of 14. A caution from the audit: 22.8% of the `gemini` entities are ambiguous, so the "inside" number is soft; and `gemini` pages labelled English are largely Italian, so English is judged on `spark` only, which is what the guard does.
  - A surviving adapter is then read on `spark/itdoc_a#60` and `chatgpt/itdoc_a`, where the original has been read with the official prompt.
  - **Outcome on the panel, 2026-10-14.** 91 pages survived the filter (of 152: 30 the original already reads, 28 left out, 3 over the cap); 90 steps in 27 minutes, peak 7.1 GB. Training loss 0.18 at step 10, 0.28 at step 30, 0.12 at step 90: it starts low and goes down slowly, as it should with a causal loss. Against `bf16_1.0_op`, a signal and not a verdict:

    | adapter | outside (25 pages) | inside (30) | English guard (14) | loops | pages under 90%, outside / guard | rule |
    |---|---|---|---|---|---|---|
    | step 30 | +0.4 [-0.3, +1.2] | +0.9 [-1.4, +3.5] | +1.7 [+0.0, +4.6] | 0, as the original | 1 -> 0 / 5 -> 3 | passes |
    | step 90 | +1.0 [+0.3, +1.8] | +3.2 [+0.6, +6.5] | +2.9 [+1.1, +5.9] | 0, as the original | 1 -> 0 / 5 -> 3 | passes |

    Step 90 is the first adapter of this project whose interval on the outside pages leaves zero out, and every outside set moves up (`spark/itdoc_a` +2.9, the four `chatgpt` sets +0.7 to +1.1). Inside, `gemini/itdoc_c` goes down (-2.9, 5 pages). It goes to the validation read (`sparkA_ft_op1_90`, `chatgptA_ft_op1_90`); 10 of those 84 pages are also in the panel.
  - **Validation read, 2026-10-14, 84 dev pages, against the original with the official prompt:**

    | set | original | step 90 | diff [95% CI] | verdict | pages lost / gained | under 90% | loops | English | Italian | accented words |
    |---|---|---|---|---|---|---|---|---|---|---|
    | `spark/itdoc_a#60` | 87.5% | 90.4% | +3.0 [+1.0, +5.1] | better | 6 / 18 | 13 -> 4 | 0 -> 1 | 85.8 -> 90.0% | 87.8 -> 90.5% | 10 of 11 -> 11 of 11 |
    | `chatgpt/itdoc_a` | 94.2% | 95.1% | +0.9 [+0.4, +1.5] | better | 2 / 12 | 0 -> 0 | 0 -> 0 | 86.1 -> 88.3% | 95.8 -> 96.5% | 66.7 -> 88.5% of 435 |

    The one new loop (`itdoc_a_0699_banknote_L5_it`) is cut by the reader at 768 tokens and costs no entity: 24 of 29 found, as the original. Tax codes go up (`cf` 58 -> 69% on spark), so do IBAN (73 -> 82%), PEC (60 -> 74%) and VAT numbers (77 -> 88%); the worst page loses 3 entities. Speed: 19.8 s per page against 18.1, on the original bf16 model with the adapter not merged. Both are dev sets, read for this decision: the claim is for these 84 pages, not yet for the test sets. Next: merge the adapter into the weights and quantize to `v8d8e8`, so that the result is a model of the size this project ships.
- **What the data can teach.** On 73 `gemini` pages BASE finds 95% of the entities in the first three lines, 96% in the body and 74% in the last three: footers are the material. On the first page of 26 `chatgpt` sets, tax codes (81%), CIG (87%) and CUP (86%) are what BASE gets wrong, one character at a time. `unità` read as `uniti` cannot be learned from these pages: the word is hardly in them.

## The models every claim is measured against

| Name | What it is | Size | s/page | Entity recall, `held/` |
|---|---|---|---|---|
| **BASE** | `zai-org/GLM-OCR`, bf16, full resolution | 2651 MB | 15.3 | 83.3% |
| **Q4** | standard 4-bit build (`mlx-community/GLM-OCR-4bit`): encoder left in bf16, decoder and embeddings at 4 bits. Ours is `v16d4e4` | 1247 MB | about 13 at 1.0x, 6.9 at 0.7x | 80.5% at 1.0x |
| **NOW** | `v8d4e4` at 0.7x: encoder at 8 bits, decoder and embeddings at 4 | 848 MB | 7.2 | 80.1% |
| **ACC** | `v8d8e8` at full resolution: encoder, decoder and embeddings at 8 bits | 1184 MB | 15.6 | 83.2% |

ACC appeared on the last diagnostic run of 2026-10-12 and changes the starting point.

Where we stand today:

- **Goal 1A is met.** ACC is lighter than Q4 (1184 against 1247 MB) and 2.6 points better (95% CI +1.3 to +4.3); it matches BASE (-0.1, CI -0.3 to +0.1). NOW is 32% lighter than Q4 and indistinguishable from it at equal resolution (+0.1, CI -0.2 to +0.4).
- **Goal 1B is half met.** At 4 bits NOW is 2.1x faster than BASE but 3.1 points worse (CI -5.5 to -1.1). ACC has BASE's accuracy but not at 4 bits, and it is not faster.
- **Goal 2 has not started.** On the 60 Italian pages of `held/`: BASE 81.6%, ACC 81.5%, NOW 77.7% (-3.9, CI -7.3 to -1.3). Italian loses more than English from the resolution cut (bf16: -3.7 on Italian, -1.7 on English).
- ACC was found on `held/`, not chosen before it. Until Phase 0 confirms it on a fresh set it is a finding, not a release.

What is established about the gap between a 4-bit model and BASE:

1. **The 4-bit decoder skips blocks of small print** at full resolution: 11 of 108 upright pages, BASE 0.
2. **It is the decoder, not the encoder.** With the encoder in bf16 and the decoder at 4 bits the skipped pages are the same (12, 11 in common). With the decoder at 8 bits there are none.
3. **A model that cannot use full resolution** loses what BASE gains there on degraded scans (BASE: +2.6 points from 0.7x to 1.0x).

So the open question for compression is no longer whether BASE quality survives quantization. It does, at 1184 MB. The question is how few decoder bits keep it, and how to get the speed back.

## Rules

These are not preferences. A result that breaks one of them is not a result.

1. **Thresholds are written before the experiment.** The targets below are fixed today. A threshold changed after seeing a number is void, and the change is recorded with its reason.
2. **Three kinds of data, never mixed.** `train` is the only data weights or calibration statistics may see. `dev` is for choosing between configurations. `test` is read **once per decision**, and each read is logged in `LEDGER.md` with date, configuration and the question asked. A configuration chosen by looking at `test` is disqualified.
3. **`hard/` is `dev`. `held/` is spent.** Eight configurations have been read on `held/` today. It stays as a regression check; it no longer decides anything. New test sets are built in Phase 0, before any new experiment.
4. **Every comparison is paired, with a 95% interval, bootstrap over pages.** "Better" means the lower bound is above zero. "Not worse" means the lower bound is above **-1.5 points**. An interval that straddles the margin means **not shown**, never pass. The remedy is more pages from a new seed, generated before looking, not a wider margin.
5. **Quality is three numbers, not one.** Entity recall, pages with a skipped block, repetition loops. A configuration that improves the first and worsens the second has failed.
6. **Speed is a clean pass.** Idle machine, one configuration after the other, median of three passes, same page set. Anything else is called indicative.
7. **Everything is rebuilt by a script** from the pinned upstream checkpoint, and every table can be recomputed from committed raw outputs.
8. **No private documents**, in any split. Real pages must come from public sources whose reuse terms are checked and written down first.
9. **English must not pay for Italian.** Every Italian gain is reported next to the English number on the same model.
10. **Negative results are published** (README and `docs/RESULTS.md`) with the same weight as positive ones. A phase that hits its kill criterion is closed, not stretched.

## Targets

| Track | Target | Stretch | Passes when |
|---|---|---|---|
| **A. Lighter than Q4** | 650 MB or less | 550 MB or less | not worse than Q4 at the same resolution on `test`; no more skipped-block pages than Q4 |
| **B. At 4 bits, faster than BASE without being worse** | not worse than BASE at full resolution, at half its time or less, with a decoder averaging 4.5 bits per weight or less | better than BASE | on `test`; skipped-block pages no more than BASE + 2 per 100 |
| **C. Italian** | better than BASE at full resolution on `test-it` | by 3 points or more | lower bound above zero on Italian entity recall; accented letters and Italian number notation not worse than BASE; English not worse than BASE |

## Phase 0 - The yardstick (2 days, blocks everything)

Nothing in Phases 1 to 4 starts before this is done.

- [x] Prove Q4 is what we say: `glmocr quant v16d4e4` is identical bit for bit to `mlx-community/GLM-OCR-4bit` (706 tensors out of 706, same file size). Done 2026-10-12.
- [ ] `dev/`: 160 pages, new seed, same generator. Replaces `hard/` for tuning (40 pages cannot separate candidates 2 points apart). Since 2026-10-13 the commissioned `spark/itdoc_a` is also `dev` (`DATASETS.md`); a fixed subsample of it has to be chosen before the first read.
- [ ] `test-it/` and `test-en/`, frozen and checksummed before any experiment:
  - 300 Italian and 150 English pages;
  - at least a third from **real public documents with a text layer** (EUR-Lex Italian/English pairs, Gazzetta Ufficiale, public tender templates), rendered and degraded by our code, ground truth from the text layer. Reuse terms checked per source;
  - the rest synthetic, from a **second renderer written independently** of `make_hard.py` (other fonts, other layouts), so a model cannot win by familiarity with one generator. Five such datasets are commissioned with `DATASET_PROMPT.md`; `itdoc_c` was designated as the test set before any of them existed, the other four go to `train` and `dev`. Delivered on 2026-10-13 by three renderers; roles in `datasets.toml`, reasons in `DATASETS.md`;
  - no pages rotated by 90 degrees (orientation is solved upstream and measured apart); the most degraded level kept but reported separately.
- [ ] Italian metrics, implemented and tested with deliberately broken inputs: entity recall on Italian pages; accented-letter recall; **Italian number notation kept** (`2.100.000,00` must not come back as `2,100,000.00`: the base model does this, rarely on our pages, 3 of 302); skipped-block rate; loop rate.
- [ ] Size the test sets: BASE against NOW must give an interval of half-width 1.5 points or less. If not, add pages now.
- [ ] Clean timing protocol scripted.
- [x] Skipped-block and loop counts printed by the scorer; every published table recomputed by `pytest`; page sets validated by `glmocr check`; a `test` set cannot be read before its tag is in the ledger. Done 2026-10-13.
- [ ] One read of `test` for BASE, Q4 and NOW: the baseline table that every later phase is compared with. Logged in the ledger.

**Exit:** baseline table committed, test checksums committed, ledger opened.

## Phase 1 - Lighter than Q4, no training (3 days)

All choices on `dev/`. One read of `test` at the end, for one configuration.

| # | Attempt | Expected | Abandon if |
|---|---|---|---|
| 1.1 | Encoder at 6 and 5 bits (it is 55% of the file) | -115 to -175 MB | recall on `dev` drops by more than 1 point, or any new skipped-block page |
| 1.2 | Quantization groups of 128 instead of 64 (69 MB of the file are per-group scales, not weights) | -34 MB | same |
| 1.3 | Per-layer sensitivity of the 24 encoder layers, then 8 bits only where it matters | -100 to -200 MB | the sensitivity ranking does not predict the measured loss on `dev` |
| 1.4 | Vocabulary pruning with a remapping table, tokenizer untouched (the first attempt renumbered the tokenizer and broke it) | -40 to -60 MB | output differs from the unpruned model on any Italian or English `dev` page |
| 1.5 | Calibrated quantization (activation-aware rounding instead of round-to-nearest), calibration pages from `train` only | same size, enables 1.1 and 1.3 at lower bits | no gain over round-to-nearest on `dev` at 4 bits |

Checked and closed: input and output vocabulary matrices are not tied (cosine 0.01), so they cannot be shared.

First pass of 1.1 and 1.2 on `hard/` (2026-10-12, table in `docs/RESULTS.md`): encoder at 4 bits fails (-3.9, CI -7.8 to -1.0); groups of 128 cost nothing; `v6d4e4g128` is 707 MB with no measurable loss. Forty pages cannot rank the 5- and 6-bit variants, because one invoice page decides them. To be redone on `dev/`.

**Exit:** one configuration at 650 MB or less that passes Track A on `test`. **Kill:** if nothing passes below 750 MB, publish NOW as the answer to 1A and stop.

## Phase 2 - How few decoder bits (3 days)

Known: decoder at 8 bits has no defect, at 4 bits it skips blocks. Everything in between is unmeasured. All at full resolution, on the invoice and footnote pages of `dev/` first (that is where the defect shows), then on all of `dev/`.

| # | Attempt | Size if it works | Abandon if |
|---|---|---|---|
| 2.1 | Decoder at 6 and 5 bits, embeddings at 4 (`v8d6e4`, `v8d5e4`) | about 970 and 900 MB | skipped pages above BASE + 2 per 100 |
| 2.2 | Embeddings and output head alone at 4 bits with the decoder at 8 (`v8d8e4`): are they part of the defect? | about 1090 MB | same |
| 2.3 | Restore single decoder layers to 8 bits in a 4-bit decoder, to find which ones carry the defect; then mixed precision at 4.5 bits per weight or less | about 880 MB | no small group of layers accounts for it |
| 2.4 | Calibrated quantization of a 4-bit decoder (from 1.5) | 848 MB | same as 2.1 |
| 2.5 | The winner of Phase 1 for the encoder combined with the winner here | target 800 MB or less at BASE quality | - |

**Exit:** the lightest configuration that passes "not worse than BASE" on `test` with no more skipped pages than BASE + 2 per 100. If its decoder averages 4.5 bits or less, Track B's quality half is done and Phase 3 supplies the speed. **Kill:** if nothing below 6 bits holds, Track B is unreachable without training and moves to Phase 4; ACC remains the answer to Track A.

## Phase 3 - Speed (3 days)

Time is reading the image (about 450 tokens/s for every variant), not writing the text. Bits do not help; fewer visual tokens do.

| # | Attempt | Abandon if |
|---|---|---|
| 3.1 | Crop empty margins before OCR | under 10% fewer tokens on `dev`, or any lost entity |
| 3.2 | Scale chosen per page from the text height measured on the image: small print and bad scans at 1.0x, the rest at 0.6x | does not beat the fixed 0.7x on both time and recall |
| 3.3 | Stop a repetition loop as soon as it starts instead of paying 3000 tokens for it (a looped page costs 30 to 55 s against 6.4 for a normal one; three such pages are 0.8 s of the 7.2 s/page average) | - |
| 3.4 | The multi-token-prediction head as a draft model (the checkpoint ships it; `mlx-vlm` has no support for this architecture) | under 1.3x on generation, or outputs differ from greedy |

Ruled out today: reading every page at two scales (+0.7 points for 3x the time; the ceiling with the answer in hand is +2.3), and detecting skipped blocks from image statistics (two estimators tried, error from 0.2x to 1.7x).

**Exit:** a clean-pass table. Today BASE quality costs 15.6 s/page (ACC); the target is 8 s/page or less.

## Phase 4 - Italian (2 weeks, the only phase with training)

Starts only when Phases 0 to 2 are closed. Training starts from the configuration that matches BASE, not from one with a known quantization defect, which training would hide instead of fixing.

- [ ] `train-it/`: at least 5000 pages. Real public Italian documents with a text layer plus synthetic pages, **from sources and seeds disjoint from every test and dev set**, checked by hash and by document identity. Degradations as in the test sets. An English share of 20% to protect rule 9.
- [ ] Teacher: BASE at full resolution where its output matches the ground truth, the ground truth itself where it does not. BASE's own omissions and rewritten numbers are not taught.
- [ ] 4.1 Low-rank adapters on the decoder and the projector, trained in bf16, merged, then quantized with the Phase 1 and 2 recipe.
- [ ] 4.2 Same, trained with quantization in the loop, if 4.1 loses its gain when quantized.
- [ ] 4.3 Only if 4.1 gains on `dev`: encoder adapters for degraded Italian scans.
- [ ] After each run: `dev` only. Italian and English side by side, and the three quality numbers.

**Exit:** one model, one read of `test-it` and `test-en`, Track C passed. **Kill:** three training runs without a lower bound above zero on `dev` Italian recall. Then the project publishes the compressed model as "equal to the base model, lighter and faster" and says plainly that it is not better in Italian.

### Probe run ahead of schedule (2026-10-12)

Asked for before Phases 0 to 2 are closed, to answer one question only: **does a fine-tune destroy the metrics we already have?** It is not a Phase 4 result and proves nothing about Italian by itself.

- Data: `itdoc_a`, the first commissioned dataset (invoices and orders, 80% Italian), levels 1 to 3 only. 160 pages to train, 60 never trained on.
- What the base model does on these pages, seen before training: it **skips the letterhead**, which is where an Italian business document keeps its VAT number, tax code and certified e-mail. That is the thing a fine-tune can plausibly fix.
- Answers: words and order from the ground truth, spacing from the model's own output (`experiments/ft_targets.py`), so the model is taught what it skips or misreads and not a new way to lay out a table. Trained on the raw ground truth, 17 to 24% of the answer tokens were layout surprises.
- Training: low-rank adapters on the decoder only (0.3% of the weights), 200 steps, bf16.
- Verdict to read: the 60 unseen pages of `itdoc_a` (does it help where it was taught?), then `hard/` and `held/` as regression checks, Italian and English apart (does it break what worked?). `held/` is used here in the only role left to it.

**Stopped before training, on purpose (same evening).** A full round of the probe costs about four hours on this machine, so the cheaper question was asked first: is what the base model loses on these pages something it does not know, or something it is not shown? On the 18 unseen `itdoc_a` pages read so far, the base model finds 78.5% of the entities. Reading the same pages in pieces, with no training (`experiments/crops.py`):

| Reading | Entities | Letterhead | Body | Footer | s/page |
|---|---|---|---|---|---|
| Whole page | 78.5% | 60% | 81% | 80% | 24.8 |
| Whole page + top and bottom bands read apart | 82.8% | 89% | 82% | 85% | 29.3 |
| Two overlapping halves, enlarged 1.4x | 89.3% | 82% | 90% | 92% | 34.3 |
| Two halves + bands | 91.3% | 93% | 91% | 92% | 38.8 |

Two halves + bands against the whole page: +12.8 points (95% CI +7.1 to +19.2), 13 pages better, none worse. VAT numbers go from 77% to 100%, IBANs from 83% to 100%, tax codes from 58% to 88%.

So most of the loss is an input problem: the model treats the letterhead as page furniture and reads it when shown alone, and it does not resolve very small print until it is enlarged. **Fine-tuning is postponed**: what would be left for it is about 9 points instead of 21, and it has to be characterised on more pages first. Caveats: 18 synthetic pages from one generator; a policy that joins several readings holds more text, so short entities have more chances to match by accident; and that version of `itdoc_a` was regenerated on 2026-10-13, so the table cannot be recomputed (raw outputs in `results/itdocval_*`). Next check: the same reading on `held/` with the 4-bit decoder, which also tells whether half pages stop it from skipping blocks.

## v1.1 - comb fields (planned on 2026-10-14, before any of its reads)

**Why.** On 50 real Italian forms (XFUND, `LEDGER.md`) v1.0.0 is not shown better than the original, and it loops twice on comb fields (digits in boxes) where the original does not; the reader cuts the loop and the rest of the form is lost.

**First fix tried: read again what loops.** A page that loops is read a second time with mlx-vlm's repetition penalty, and `engine.keep_retry` keeps the second reading if it does not loop, else the one with more text before its loop (rule fixed before measuring, no ground truth). It costs time only on pages that loop. Training on comb fields comes second, with pages we draw ourselves, only if this is not enough.

**Where it is chosen.** `external/xfund_it_train` (149 real forms, `dev`): BASE and v1.0.0 read once (`xfundtr_bf16_1.0_op`, `xfundtr_v8d8e8op1_1.0_op`), then the looping pages of v1.0.0 read again with penalties 1.05, 1.1 and 1.2 (`experiments/loop_retry.py`, tags `xfundtr_v8d8e8op1_1.0_op_rp105` and so on). The penalty kept is the one with the highest entity recall on all entities; on a tie, the smaller. It must also leave the synthetic dev pages no worse: the retry touches only pages that loop, so on `spark/itdoc_a#60` and `chatgpt/itdoc_a` it is checked on those pages alone.

**Outcome on `external/xfund_it_train`, 2026-10-14** (149 forms, against BASE `xfundtr_bf16_1.0_op`; loops cut or not):

| | loops | entities | against BASE [95% CI] | extra time |
|---|---|---|---|---|
| v1.0.0 | 6 (BASE: 1) | 78.5% | +1.2 [-1.1, +3.1] | |
| retry at 1.05 | 5 | 79.0% | +1.7 [-0.2, +3.5] | 0.8 s per form |
| **retry at 1.1** | **1** | **80.2%** | **+2.9 [+1.1, +4.2]** | **1.0 s per form** |
| retry at 1.2 | 2 | 79.6% | +2.3 [+0.2, +4.0] | 0.9 s per form |

The penalty kept is **1.1**, by the rule above. Without retry, v1.0.0 gains 377 entities on the 144 forms where it does not loop and loses 238 on the 5 where it loops and BASE does not. One form (`it_train_120`) loops at the same place at every penalty: retry does not reach it. On the synthetic dev pages v1.0.0 never loops, so retry changes nothing there. Accented words written right: BASE 90.6%, v1.0.0 88.8% of 1425. Looked into: the gap is the text lost in loops. Counted per word with repeats (1426): BASE 88.5%, v1.0.0 87.2%, v1.1 90.1%; on the 144 forms where v1.0.0 does not loop, BASE 89.8%, v1.0.0 91.5%. The errors left are shared: both models write "Si" for "Sì" (36 times) and "altresi" (12).

**Then one test read**, question written first: v1.1 (v1.0.0 weights + retry) against BASE on `external/xfund_it_val`. That set was read once for v1.0.0, and it is what pointed at comb fields; the setting is chosen on another split, and the second read is said as such.

**Outcome.** Released as v1.1.0 on 2026-10-14: on `external/xfund_it_val` +3.1 [+1.3, +5.9] over BASE, loops 2 -> 0 (`LEDGER.md`). On the OmniDocBench subset Chinese is worse (-4.8) with or without the retry, English not shown either way.

## After v1.1 (2026-10-14 and 15): fine-tuning runs op2 to op4, new page sets, v1.2.0

**Released as v1.2.0 (2026-10-15): the v1.1 weights with PDFs read from their text layer** (`glmocr/pdf.py`, idea and checks from Indra OCR by Datalab). None of the fine-tuning runs below was released. Outcome in `docs/RESULTS.md`, "Release v1.2.0", and `LEDGER.md`.

**Rule on languages, set by the owner on 2026-10-14.** Italian and English must not lose. Chinese and non-European languages may, even badly. Other European languages are watched: a large loss there is an alert, not a blocker. The format of tables (markdown or plain) does not matter.

**Pilot `op2`, a hint and not a release** (owner's go, 2026-10-14). Data: the 91 pages of `op1` (Italian and English, so that what op1 learnt is kept) + 35 comb-field pages (12 clean F24, 11 degraded F24, 12 degraded printable forms; `external/ft_tr_cb`, teacher `fttr_cb_bf16_1.0_op`, answers by `ft_targets.py`) = `work/ft/train_op2.jsonl`, 126 pages, one pass, lr 1e-5, rank 8. Trained on the Mac mini (this machine stopped it for swap at step 10). Read on dev only, raw (no retry) against raw v1.0.0: XFUND Italian train (loops and entities), `spark/itdoc_a#60` and `chatgpt/itdoc_a` (Italian and English must not lose), `external/gu_dev`. Queue: `work/pilot_op2.sh` on the mini. What would make it a v1.2: fewer loops on comb fields with Italian and English not worse on dev; then the test reads, with fresh test sets.

**New page sets.** `external/gu_dev` and `external/gu_test` (Gazzetta Ufficiale, real Italian print, text layer as ground truth, `generators/make_gu.py`): the fresh Italian test set that XFUND val, read twice, no longer is. `external/xfund_{de,es,fr,pt}_val`: the European guard, question in `LEDGER.md`, reads queued on this machine (`work/eu_reads.sh`, BASE, v1.0.0, retry).

**Night of 2026-10-14 to 15: what v1.1 still reads badly** (`LEDGER.md`, all test sets of that night). Not worse than BASE on every Italian and European set but one; the losses that remain are four kinds of page: blanks drawn as rows of dots or underscores (it writes the row and stops: DocAtlas, the Gazzetta, OmniDocBench), tick boxes (a run of empty boxes until the token limit: XFUND, `it_train_136`), near-identical sentences one after another (`itdoc_10_adversarial`, `itdoc_14_scientific_0081`), amounts of tens of millions (written "29.467,745,94", `itdoc_10_adversarial`). Restoring the loop guard to let dot rows run longer recovered nothing: the model itself stops or loops there.

**Data for `op3`; training started with the owner's go (2026-10-15, on the Mac mini, `work/pilot_op3.sh`).** The 126 pages of `op2` + 142 pages chosen for those four weaknesses (`external/ft_tr_op3`, `experiments/ft_data.py --per-set 60`): 60 real Italian documents of DocAtlas with blanks to fill (45), tick boxes (20) or amounts of a million or more (5), half of them scanned (`make_docatlas.py --train`, documents not reserved for test); 30 Gazzetta pages with amounts of millions (61) and dot leaders (25), from issues in neither gu_dev nor gu_test (`make_gu.py --train`); 52 pages of `itdoc_13_realistic` levels 1 to 3, whose sentences repeat with small changes. Answers by `ft_targets.py` from the BASE read `fttr_op3_bf16_1.0_op`. `itdoc_14_scientific` stays out of training, by the owner's choice. Measured afterwards on dev only (XFUND Italian train, `spark/itdoc_a#60`, `chatgpt/itdoc_a`, `gu_dev`), and on the panel pages held back from each new set; the test sets of that night have all been read once and would be read a second time.

**How `op3` is judged, written on 2026-10-15 before any of its reads was looked at.** Trained 79 minutes on the Mac mini, 248 pages (5 over 4608 tokens left out), 253 steps; loss 0.36 at step 10 down to 0.05 to 0.2, no collapse to zero; no held-back, test or dev page in its data (checked by id, document and issue). Every comparison is op3 with the retry (`_rp110`) against v1.1 (op1 with the retry), paired bootstrap over pages, rule 4:
1. Must not lose, on dev: `xfundtr`, `gudev`, `sparkA` (Italian and English), `chatgptA`. Each lower bound above -1.5.
2. Must gain where it was trained to, on the 30 held-back panel pages (`panelda`, `panelgu`, `panelre`, pooled): difference above 0 and lower bound above -1.5; reported with the counts behind it: pages that stop early (skipped block), loops, amounts of a million or more read as written.
Both, or op3 is not a candidate. A candidate gets one test read, question written first: `external/gu_test` (never read); the sets of the night of 2026-10-14 would be second reads, said as such.

**Outcome of `op3` (2026-10-15): not a candidate, stopped.** Criterion 1 fails: on XFUND Italian train op3 with the retry is worse than v1.1, -2.1 [-3.1, -1.1] (76 pages lost, 26 gained; printed labels and options, not filled-in values); Gazzetta dev -1.7 [-6.1, +2.0] and chatgpt A -2.1 [-6.6, +0.4] not shown; spark A not worse. Criterion 2: no gain on the DocAtlas panel (-0.2 [-0.6, +0.0]); the Gazzetta and realistic panels were stopped unread, by the owner, as they could not change the outcome. What op3 does wrong, measured: on XFUND it writes as many characters as v1.1 (364 thousand) in fewer lines (5688 against 6482): it joins lines and moves fields, which breaks the multi-word printed labels and options (entity types question and other); filled-in values hold (+4). First reading of the cause, said too fast and wrong: "the DocAtlas ground truth leaves out the box options". It does not: they are there ("Lunedì Martedì ... ☐ ☐ ☐"); a word check took the teacher's "Lunedi" for a missing word. What the DocAtlas ground truth does have: markdown escapes (`\_` for every underscore of a blank, 3316 on 16 of the 58 pages), whole forms on one line (5 pages), and the order of the DOCX instead of the order on the page (4 pages under 0.7 of the teacher's words in order). The Gazzetta and realistic answers have none of these. Likely cause, not proven: answers whose order and lines differ from the page. Before any `op4`: answers with no markdown escape, in the order of the page; a gate in `ft_targets.py` for pages whose ground truth lacks what the teacher reads (written, calibrated on op1, op2 and op3 data: it leaves out none of op1's 91 pages nor op2's 35, and 12 DocAtlas pages); and, to know rather than guess, an ablation: op2 data + the Gazzetta and realistic pages only. A new training needs a new go.

**`op4`, an ablation (owner's go, 2026-10-15), written before training.** Question: does the loss of op3 on XFUND Italian train come from its DocAtlas pages? Data: op3's data without DocAtlas: the 126 pages of op2 + 17 Gazzetta + 52 realistic pages (answers from the same teacher read, `ft_targets.py` with its new gate) = `work/ft/train_op4.jsonl`, 195 pages; same settings as op3 (one pass, lr 1e-5, rank 8). Read with the retry, against v1.1, on criterion 1 of op3 only (XFUND Italian train, Gazzetta dev, spark A, chatgpt A), XFUND on the Mac mini and the other three here, for time. Reading: if XFUND is not worse (lower bound above -1.5) where op3 was worse, DocAtlas as built is the cause; if it is still worse, the cause is elsewhere (the Gazzetta or realistic answers, or the amount of training), and DocAtlas is not cleared or blamed. Either way op4 is not a release candidate by itself: it was built to answer this question.

**Outcome of `op4` (2026-10-15).** Against v1.1, with the retry on both sides: XFUND Italian train -2.3 [-3.2, -1.3] worse, as op3 (-2.1); Gazzetta dev -0.6 [-3.9, +2.0] and spark A -0.5 [-1.5, +0.4] not shown. So DocAtlas is not the cause of the XFUND loss. Looking back: op2 itself, read the same way, is -2.1 [-3.4, -1.0] worse than v1.1 on XFUND Italian train; it looked harmless on 2026-10-14 because it was compared without the retry, with v1.0.0 (-0.9, not shown). op3 and op4 inherit the loss from what op2 adds to op1: the 35 comb-field pages (F24 clean and degraded, printable forms) or the training itself. The next isolation, if wanted: op1's data + the Gazzetta and realistic pages, without op2's 35. DocAtlas answers cleaned on 2026-10-15: escapes removed, pages out of reading order or flattened into one line left out by `ft_targets.py` (47 of 60 kept). Answer order does not explain op2's loss either: measured on letters and digits, op2's printable forms are in order (1.00) and its F24 pages close to it (0.89 to 0.91); the 0.72 first seen was the spacing of codes read box by box.

**OmniDocBench, plan agreed on 2026-10-15.** Where v1.1 stands, on `omnidoc_200` (text blocks only): the loss is not mostly the model's. (1) We read a page whole at 150 dpi with 3000 tokens; GLM-OCR's own pipeline (SDK `zai-org/GLM-OCR`, Apache-2.0) detects regions with PP-DocLayoutV3 and reads each at 200 dpi with a prompt per kind (text, table, formula) and 8192 tokens. Dense multi-column pages show it: newspapers come back at 0.44 to 0.88 of their text, Chinese newspapers loop on 4 of 5, for the original model as much as for ours. (2) The benchmark's overall score averages text, tables (TEDS) and formulas (CDM); a whole page read with "Text Recognition:" was scored on text only. (3) Fine-tuning cost some kinds of page against the original (text edit x100): English books 6.0 -> 9.9, English academic papers 5.4 -> 7.9, Chinese books 0.2 -> 17.3, Chinese research reports 7.0 -> 16.6; it gained on English newspapers, textbooks, magazines and exams.
Decisions of the owner, same day: **Chinese is in scope** for OmniDocBench (the language rule's "Chinese may lose" no longer holds there); formulas scored by edit distance for now, said as such, CDM later.
- **Phase 0, measurement (done 2026-10-15).** `omnidoc_200` spent; `omnidoc_dev` and `omnidoc_test2`, 200 pages each, disjoint, in the benchmark's language proportions; `omnidoc_eval.py --full` scores tables and formulas too, with `overall_edit`.
- **Phase 1, a region pipeline, no training.** PP-DocLayoutV3 (Apache-2.0, transformers weights) for regions, then each region read with its own prompt, after the SDK's logic, credited. Measured on dev, BASE and v1.1, whole page against regions. Stop if regions do not beat the whole page on dev.
- **Phase 2, recover what fine-tuning cost (needs a go).** Rehearsal pages whose answer is the original's own good reading (books, papers, reports, tables of contents, Chinese pages), every answer through `ft_targets.py`'s checks; first, isolate op2's XFUND loss.
- **Phase 3, one test read** on `omnidoc_test2`, question written first: candidate against the original, both through the same pipeline.

**GGUF track (agreed 2026-10-14, after the pilot).** llama.cpp supports GLM-OCR since PR #19677 (model + `mmproj`; `--flash-attn off` needed for now). The fused bf16 checkpoint (`work/ft/op1_90_merged`) has the upstream layout, so `convert_hf_to_gguf.py` should take it as is. A GGUF Q8_0 is a different model from the one measured (other quantization, other image preprocessing): it is read on the same sets with the official prompt and the same retry, and published only if Italian and English do not lose against the MLX release. Then, maybe, an application on top (a validated Italian entity extractor first; an app, or a paperless-ngx OCR backend, after).

**Open.** The F24's reuse terms before publishing anything trained on it. Photos of the printed forms (X01, X02 as test). A JSON-extraction track (CORD for it). The old private repository to delete (needs `gh auth refresh -s delete_repo`).

## Later

- **A larger OCR as a teacher, on real images with no transcription** (the photo set, real documents): filtered, since a teacher errs too (keep what two teachers agree on). An open model as teacher: the terms of most API models forbid training on their output.
- **GRPO or DPO on small comb-field crops**, with rewards that need no reference: the check character of a tax code, an IBAN or a VAT number read back is right or wrong, and an output loops or not. Several samples per crop: slow on this machine, possible on crops.

## Phase 5 - Release

- [x] Final table on `test`: BASE and one released configuration, `v8d8e8` with adapter `op1` fused in (v1.0.0, 2026-10-14; `docs/RESULTS.md`, `LEDGER.md`). Q4 was compared with `v8d8e8` on `test` on 2026-10-13 and not shown worse or better; the "small" configuration is not released, because 0.7x loses small print.
- [x] Model card with the tables, the limits, the recipe and MIT attribution to the upstream model (`docs/MODEL_CARD.md`); the weights rebuild by script from the pinned checkpoint and the published adapter.
- [x] Real scans: 100 SROIE receipts, not worse than the original (`LEDGER.md`).
- [x] Code license: MIT.
- [ ] GGUF only if `llama.cpp` supports this architecture, and measured on the same `test`: a second runtime gets no credit from the first one's numbers.

## Not doing

- **Tuning on `test`**, including "just checking" a candidate there.
- **Private or client documents**, for training or testing.
- **Claims about real scans from synthetic pages alone.** Until Phase 0 adds real pages, every number carries the word synthetic.
- **A new architecture.** Replacing the encoder or the decoder is another project.
- **Reopening the choice of base model.** PaddleOCR was compared with GLM-OCR on our documents before this repository started, and GLM-OCR won: that is why the work is on GLM-OCR. Public leaderboards on Chinese and English pages (for example Real5-OmniDocBench, where GLM-OCR is fifth among models under one billion parameters) do not change that.
- **Layer removal without training.** It goes inside Phase 4 if there is budget left, not before.
- **Reporting the best seed, the best page set or the best scale** found after the fact.
