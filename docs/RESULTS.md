# Results

The full record of the measurements behind each release, newest first, moved here from the README on 2026-10-15 unchanged. Every number in these tables is recomputed from `results/` by the test suite (`tests/test_tables.py`, and `tests/test_release.py` where the page sets are on the machine). Each test-set read had its question and threshold written in `LEDGER.md` before it ran.

## Release v1.2.0

Same weights as v1.0.0 and v1.1.0. What is new is how a PDF is read: `glmocr infer file.pdf` takes each page's own text layer where it passes four checks, and reads the page with the model only where it does not (`glmocr/pdf.py`, mode `auto`; `--mode ocr` always uses the model). A layer is not trusted when it is an OCR layer laid invisibly over a scan, when its font map is broken, when its text has no ink under it, or when its content order is not a reading order.

The idea and the checks come from **Indra OCR** by Datalab ([github.com/datalab-to/indra](https://github.com/datalab-to/indra), `indra_ocr/textlayer.py`, Apache-2.0; `NOTICE`). Two changes are ours: Latin text whose font map comes out as letters of another script is not trusted (the running head of the Gazzetta Ufficiale, set in small caps, reads "Gൺඓඓൾඍඍൺ Uൿൿංർංൺඅൾ"), and characters beyond the BMP, which pdfium returns in two halves, are joined before they are checked. Only Indra's code is used; its model weights are under another license and are neither included nor used.

| Page set | Role | Pages | Original | v1.1.0 | **v1.2.0 (auto)** | v1.2.0 - v1.1.0 [95% CI] | Verdict |
|---|---|---|---|---|---|---|---|
| Omni Parse Bench subset, real PDFs, text tests passed | test | 80 | 56.0% | 62.5% | **76.1%** | **+12.2 [+5.8, +19.6]** | better |
| same, tests not taken from the PDF itself | test | 49 | 62.7% | 70.9% | **80.9%** | +8.2 [+2.0, +16.3] | better |
| `itdoc_14_scientific`, Italian PDFs, entities (second read) | test | 100 | 96.3% | 97.3% | **100.0%** | +2.7 [+0.9, +5.3] | better |

- **Omni Parse Bench** (Datalab, [github.com/datalab-to/omni_parse_bench](https://github.com/datalab-to/omni_parse_bench)): 80 PDF pages, 8 from each of ten suites of real documents (filled forms, multilingual PDFs, multi-column, old scans, tiny text, tables), scored by the benchmark's own scorer on its 184 text tests. 29 of the 80 pages pass the checks and are read from their layer, none of them worse than v1.1.0's reading; the other 51 are v1.1.0's readings. Some tests are derived from the PDF itself and favour the layer: on the others the gain is +8.2. The same set measures v1.1.0 against the original: **+6.9 [+2.2, +12.4]**, no page worse, English alone +4.0 [+0.0, +9.9]. The absolute numbers are not those of the benchmark's own table (GLM-OCR 71.0 on text there), which runs the vendor's pipeline: layout detection, then each region read at 200 dpi. Here a page is read whole at 150 dpi.
- **Speed.** A page read from its layer takes about 0.01 s and loads no model; a PDF that is all born digital never loads the model.
- **Limits.** 80 pages and 184 tests are a hint, not a ranking, and few of them are Italian. The scientific set is the best case by construction: its PDFs and its ground truth come from the same source. Text drawn as paths or inside images is not in a layer and is lost on a page read from its layer (a scanned stamp on a digital page). A page whose content order fails the check goes to the model even when its text is sound: on the Omni Parse Bench subset, 19 of 80.

**Also measured since v1.1.0, same weights** (`docs/LEDGER.md`): XFUND val in German +2.5 [+0.0, +5.0], Spanish +3.8 [+2.3, +5.6], French +4.0 [+2.4, +5.7], Portuguese +3.2 [+2.1, +4.5] over the original, all better; 40 real Italian documents of DocAtlas not worse, clean -0.1 [-0.8, +0.4] and scanned -0.3 [-1.0, +0.2]. One alert: on 10 adversarial synthetic pages v1.1.0 is worse, -5.3 [-11.3, -0.7], mostly amounts of tens of millions written with a comma as thousands separator ("29.467,745,94"). On the 113 amounts of a million or more in the Gazzetta Ufficiale dev set it reads more of them right than the original, 108 against 105.

## Release v1.1.0

Same weights as v1.0.0. What changes is how a page is read: a page that falls into a repetition loop is read a second time with a repetition penalty of 1.1 (mlx-vlm's), and the second reading is kept unless it loops too and returns less text (`engine.keep_retry`, fixed before any retry was measured). `glmocr infer` does it by default. The penalty was chosen on the 149 forms of the XFUND Italian training split, used as a dev set and never trained on; 1.05, 1.1 and 1.2 were tried (`docs/ROADMAP.md`).

| Page set | Role | Pages | Original | v1.0.0 | **v1.1.0** | v1.1.0 - original [95% CI] | Verdict | Loops |
|---|---|---|---|---|---|---|---|---|
| XFUND Italian forms, val (real scans) | test | 50 | 74.6% | 74.9% | **77.7%** | **+3.1 [+1.3, +5.9]** | better | 2 / 3 / **0** |
| same, filled-in values only | test | 50 | 76.9% | 74.5% | 77.9% | +1.1 [-0.9, +3.2] | not worse | |
| XFUND Italian forms, train | dev | 149 | 77.3% | 78.5% | 80.2% | +2.9 [+1.1, +4.2] | better | 1 / 6 / 1 |
| OmniDocBench v1.5 subset, English (text edit distance x100, lower is better) | test | 140 | 7.6 | 8.1 | 7.9 | -0.3 [-2.7, +1.7] | not shown | |
| OmniDocBench v1.5 subset, Chinese | test | 45 | 14.8 | 19.6 | 19.7 | **-4.8 [-10.6, -0.9]** | **worse** | 5 / 7 / 5 (all 200 pages) |

- **Real Italian documents, measured.** XFUND (Xu et al., 2022) holds real forms, filled in by hand or by machine and scanned; entities are its annotated segments of 2 characters or more. v1.0.0 alone was not shown better there: it fell into loops on comb fields (one character per box, as in tax codes) on 2 of 50 forms and lost the rest of each form. The retry reads both again in full.
- **Chinese pays.** On OmniDocBench (200 pages of the benchmark, 150 English and 50 Chinese, scored by its own evaluator on text blocks; formulas and tables left out) the model is not shown to keep English and is worse on Chinese, both as v1.0.0 and as v1.1.0. No training page was Chinese. On 2 Chinese pages the model writes a table as markdown, as it was trained to, and the evaluator files it under tables; that explains part of the loss, but the split was made after seeing the result. For Chinese documents, use the original.
- **Cost.** A second read only where a page loops: 22.5 s per form on average against 20.7 for v1.0.0 and 21.5 for the original (Mac mini M4).
- **Limits.** XFUND val had been read once for v1.0.0, and that read is what pointed at comb fields; this second read is of a configuration chosen on another split, and is said as such in `docs/LEDGER.md`. The next model change needs a fresh test set. Accented words written right on the dev split: 88.5% for the original, 90.1% for v1.1.0; the drop first seen with v1.0.0 was the text lost in loops (on the forms where v1.0.0 does not loop, 89.8% against 91.5%). Both models write "Si" for "Sì". XFUND is CC BY-NC-SA 4.0, used for evaluation only; its outputs are not in git, nor are those of OmniDocBench, whose pages include copyrighted newspapers and books.

## Release v1.0.0

`v8d8e8_op1` = a LoRA adapter (rank 8, decoder only, 90 steps on 91 synthetic Italian and English pages) fused into the original weights, then quantized: vision encoder, decoder, embeddings and output head at 8 bits. Measured on an M4 with 16 GB, official prompt, greedy decoding. "Original" is `zai-org/GLM-OCR` in bf16.

| Page set | Role | Pages | Original | **v1.0.0** | Difference [95% CI] | Verdict | Pages under 90% of the text | Loops |
|---|---|---|---|---|---|---|---|---|
| `spark/itdoc_b`, Italian | test | 95 | 88.7% | **92.4%** | **+3.7 [+2.4, +5.1]** | better | | |
| `spark/itdoc_b`, English | test | 25 | 86.8% | **93.9%** | **+7.1 [+4.1, +10.5]** | better | | |
| `spark/itdoc_b`, all | test | 120 | 88.3% | **92.7%** | +4.4 [+3.2, +5.8] | better | 25 -> 3 | 0 -> 2, cut, no entity lost |
| `chatgpt/itdoc_c` | test | 24 | 93.0% | 93.7% | +0.7 [+0.0, +1.4] | not worse | 0 -> 0 | 0 -> 0 |
| SROIE 2019, real receipts (English) | test | 100 | 92.0% | 92.0% | +0.0 [-1.2, +1.5] | not worse | 2 -> 0 | 0 -> 0 |
| `spark/itdoc_a` | dev | 60 | 87.5% | 90.0% | +2.5 [+0.7, +4.6] | better | 13 -> 5 | 0 -> 0 |
| `chatgpt/itdoc_a` | dev | 24 | 94.2% | 94.7% | +0.5 [-0.1, +1.1] | not worse | 0 -> 0 | 0 -> 0 |

| | Size | Peak memory | Generation | s/page (`spark/itdoc_b`) |
|---|---|---|---|---|
| Original (bf16) | 2651 MB | 3.16 GB | 65 tok/s | 17.8 |
| **v1.0.0** | **1184 MB** | **2.10 GB** | **72 tok/s** | 18.0 |

- **Italian up, English up, on a renderer no training page came from.** Entities are amounts, dates, IBAN, VAT numbers, tax codes, CIG and CUP codes, PEC addresses, contract numbers. On `spark/itdoc_b` tax codes go from 60% to 72%, IBANs from 72% to 84%, PEC addresses from 60% to 79%, VAT numbers from 73% to 87%. Most of the gain is text the original leaves out: letterheads and footers.
- **Not worse where the original was already good** (`chatgpt/itdoc_c`, 93%), and **not worse on real scanned receipts** (SROIE), which no training page resembles.
- **Same size and speed as the plain 8-bit compression** (`v8d8e8`): fine-tuning adds nothing at inference. The questions and thresholds of every test read were written in `docs/LEDGER.md` before the read; each test set was read once for this model.
- **Limits.** At this release every Italian page measured was synthetic: three renderers (`gemini` for training, `spark` and `chatgpt` for evaluation), commissioned for this work. Real Italian documents were measured from v1.1.0 on. The SROIE row is a 100-receipt subset, read on a second M4. The `dev` rows guided the choices; only the `test` rows are claims.

What did not work on the way, and why, is in `docs/ROADMAP.md`: three void fine-tuning runs (mlx-vlm 0.7.6 trains GLM-OCR without a causal mask, [mlx-vlm#2509](https://github.com/Blaizzy/mlx-vlm/issues/2509)), and a prompt that differed from the official one by four tokens.

## Compression study

The work that chose the 8-bit configuration. Measured on 120 held-out synthetic pages, M4 with 16 GB, with mlx-vlm's prompt (every run before 2026-10-14; the official prompt moves these numbers by less than a point and changes no comparison, `docs/ROADMAP.md`).

| | Size | s/page | Peak memory | Entity recall | Against the original |
|---|---|---|---|---|---|
| Original (bf16), full resolution | 2651 MB | 15.3 | 3.16 GB | 83.3% | |
| Standard 4-bit build, full resolution | 1247 MB | 12.8 | 2.19 GB | 80.5% | -2.7 [-4.3, -1.4] |
| **`v8d8e8`, full resolution** ("accurate") | **1184 MB** | 15.6 | 2.10 GB | **83.2%** | **-0.1 [-0.3, +0.1]** |
| **`v8d4e4` at 0.7x** ("small and fast") | **848 MB** | **7.2** | **1.74 GB** | 80.1% | -3.1 [-5.5, -1.1] |

- **`v8d8e8` is the original model at 45% of its size.** It is also lighter than the standard 4-bit build and 2.6 points better than it (95% CI +1.3 to +4.3). It is not faster than the original. It was confirmed on two test sets of other renderers (`docs/LEDGER.md`), and it is the base of the release.
- **`v8d4e4` at 0.7x is 3.1x smaller and 2.1x faster** than the original, and loses 3.1 points of entity recall. It matches the standard 4-bit build (-0.4, CI -2.5 to +1.2) at two thirds of the size and 56% of the time.
- **Bits belong in the decoder, not in the encoder.** The vision encoder at 8 bits is indistinguishable from bf16. The decoder at 4 bits is what costs: it silently skips blocks of small print (invoice detail lines, footnotes). The standard build does the opposite, leaving the encoder in bf16 and squeezing the decoder.
- **The 0.7x scale is not safe on dense small print.** On pages from another renderer, with print 14 pixels high, the same configuration loses 34 points (see the pilot below).
- Everything here is measured on **synthetic pages**.

Naming: `v8d4e4` = vision encoder at 8 bits, decoder at 4, embeddings and output head at 4 (16 = left in bf16). The standard 4-bit build is `mlx-community/GLM-OCR-4bit`; `glmocr quant v16d4e4` reproduces it bit for bit (706 tensors out of 706), so `v16d4e4` in the tables is that build. The upstream checkpoint also carries a multi-token-prediction head (436 MB) that inference does not use; quantized variants drop it.

## Test sets

Both are generated by `generators/make_hard.py`: 8 page templates (prose, two columns, table, form, invoice, footnotes, bilingual, numbered clauses), Italian and English, 5 degradation levels, with exact ground truth and a list of entities to recover (amounts, dates, IBAN, VAT number, tax code, CIG, contract number, handwritten amounts).

| Set | Pages | Entities | Role |
|---|---|---|---|
| `hard/` | 40 | 483 | Tuning set: every choice in this repository was first made here |
| `held/` | 120 | 1434 | Held out: other seed, three pages per template and level, variable layouts (row counts, footnote size and count). Generated after `v8d4e4` at 0.7x had been chosen. Eight configurations were read on it, each once; every read is in `docs/LEDGER.md` |

Levels: L1 clean, L2 light scan, L3 bad scan at low dpi with a handwritten correction, L4 stamp, warp and noise, L5 tiny font, heavy noise, low contrast, fold shadow, and half of the pages rotated by 90 degrees.

The metric is **entity recall**: an entity counts if its exact string, whitespace removed, appears in the output. Confidence intervals are a bootstrap over pages (2000 resamples), because a failure takes out a page or a block, not a single entity. Differences between two configurations are paired on the same pages.

Two limits of this metric. It is recall only: a short entity can also match by accident, so it is reported next to the two defect counts (skipped blocks, loops) and never alone. And the handwritten amounts of levels 3 to 5 are entities but are not in the ground-truth text, so they count for recall and are invisible to the character error rate and to the skipped-block count.

The commissioned datasets (`train`, `dev` and `test`, three renderers, 39000 training pages) are described in `docs/DATASETS.md`; they live outside this repository.

## Results on the held-out set

`held/`, 120 pages, greedy decoding, one run after the other on an otherwise idle machine.

| Config | Size | Entity recall [95% CI] | L1 | L2 | L3 | L4 | L5 | Pages with a skipped block | Loops | s/page | gen tok/s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 @1.0 | 2651 MB | 83.3% [77.7, 88.2] | 100 | 100 | 97 | 89 | 35 | 0/108 | 4 | 15.3 | 68 | 3.16 |
| **v8d8e8 @1.0** | 1184 MB | 83.2% [77.6, 88.2] | 100 | 100 | 97 | 89 | 35 | 0/108 | 4 | 15.6 | 74 | 2.10 |
| v16d4e4 @1.0 (standard 4-bit) | 1247 MB | 80.5% [74.6, 85.8] | 99 | 99 | 93 | 86 | 30 | 12/108 | 3 | 12.8 | 99 | 2.19 |
| bf16 @0.7 | 2651 MB | 80.6% [74.8, 85.8] | 100 | 100 | 93 | 82 | 34 | 1/108 | 4 | 10.9 | 67 | 3.07 |
| v8d4e4 @1.0 | 848 MB | 80.6% [74.7, 85.9] | 99 | 99 | 93 | 86 | 31 | 11/108 | 3 | 13.5 | 97 | 1.77 |
| v8d4e4 @0.85 | 848 MB | 81.0% [75.3, 86.1] | 99 | 99 | 94 | 86 | 32 | 10/108 | 1 | 9.6 | 109 | 1.77 |
| **v8d4e4 @0.7** | 848 MB | 80.1% [74.1, 85.6] | 99 | 100 | 92 | 85 | 30 | 3/108 | 3 | 7.2 | 131 | 1.74 |
| v8d4e4 @0.6 | 848 MB | 79.0% [72.9, 84.6] | 100 | 96 | 93 | 83 | 28 | 1/108 | 3 | 5.7 | 147 | 1.74 |

"Skipped block" = an upright page whose output holds less than 90% of the reference characters. "Loops" = pages whose output is a repetition loop (see failure modes). Both are printed by `glmocr score`. Scale 0.7 means the page image is resized to 70% per side before OCR (1380 visual tokens instead of 2788).

Paired differences, same 120 pages:

| Comparison | Difference [95% CI] |
|---|---|
| v8d8e8 @1.0 vs bf16 @1.0 | -0.1 [-0.3, +0.1] |
| v8d8e8 @1.0 vs v16d4e4 @1.0 | +2.6 [+1.3, +4.3] |
| v16d4e4 @1.0 vs bf16 @1.0 | -2.7 [-4.3, -1.4] |
| v8d4e4 @1.0 vs v16d4e4 @1.0 | +0.1 [-0.2, +0.4] |
| v8d4e4 @0.7 vs v16d4e4 @1.0 | -0.4 [-2.5, +1.2] |
| v8d4e4 @0.7 vs bf16 @1.0 | -3.1 [-5.5, -1.1] |
| v8d4e4 @1.0 vs bf16 @1.0 | -2.6 [-4.3, -1.2] |
| bf16 @0.7 vs bf16 @1.0 | -2.6 [-5.4, -0.1] |
| v8d4e4 @0.7 vs bf16 @0.7 | -0.5 [-2.2, +1.4] |
| v8d4e4 @0.6 vs v8d4e4 @0.7 | -1.1 [-2.8, +0.4] |
| v8d4e4 @0.85 vs v8d4e4 @0.7 | +0.9 [-0.7, +2.8] |
| v8d4e4 @1.0 vs v8d4e4 @0.7 | +0.5 [-1.2, +2.6] |

What this shows:

- **The original model gains from full resolution; a 4-bit decoder does not.** bf16 gains 2.6 points going from 0.7x to 1.0x, on the degraded levels. v8d4e4 gains nothing (+0.5, inside the interval), because at higher resolution it starts skipping blocks: 11 pages at 1.0x against 3 at 0.7x, while bf16 at 1.0x skips none.
- **The cause is the decoder's 4 bits, not the encoder's 8.** Leaving the encoder in bf16 changes nothing: v16d4e4 skips 12 pages, 11 of them the same as v8d4e4, and the two differ by 0.1 points. Raising the decoder to 8 bits removes the defect entirely: v8d8e8 skips no page and is within 0.1 points of bf16, with an interval of [-0.3, +0.1].
- **With a 4-bit decoder recall is flat from 0.6x to 1.0x** and time grows from 5.7 to 13.5 s/page. 0.7x is the point where nothing measurable is lost against any higher scale; below 0.6x recall drops (71.6% at 0.5x on `hard/`).
- **Skipped blocks happen on two templates only**: invoices and footnotes, the ones with the smallest print. No prose, table, form, two-column, bilingual or clause page lost a block at any setting, with one exception: a bilingual page at L5, the unreadable level, for the standard 4-bit build.
- **L5 is mostly unreadable for every variant**, the original included.

These two configurations were not both chosen in advance: `v8d4e4` at 0.7x was fixed on the tuning set before `held/` existed, while `v8d8e8` at 1.0x was run on `held/` as a diagnostic and turned out to match the original. It still has to be confirmed on a set it was not found on (see `docs/ROADMAP.md`).

By language, against bf16 at full resolution: v8d8e8 -0.1 on Italian and 0.0 on English; v8d4e4 at 0.7x -3.9 on Italian (CI -7.3 to -1.3) and -2.4 on English (CI -5.7 to +0.3). The resolution cut hurts Italian pages more.

## Pilot on the commissioned sets

First reads of pages from renderers other than ours (2026-10-13, environment of `requirements.lock`): the first 60 pages of `spark/itdoc_a` and the 24 pages of `chatgpt/itdoc_a`, both `dev` sets (`docs/DATASETS.md`). A pilot to size the measurements, on few pages; the tests do not recompute it, because the page sets are not in this repository.

| Config | `spark/itdoc_a#60` | against bf16 @1.0 | `chatgpt/itdoc_a` (24) | against bf16 @1.0 | s/page, spark / chatgpt |
|---|---|---|---|---|---|
| bf16 @1.0 | 88.3% [84.8, 91.2] | | 94.4% [92.2, 96.3] | | 18.5 / 43.4 |
| **v8d8e8 @1.0** | 88.3% [84.9, 91.2] | 0.0 [-0.3, +0.3] | 94.4% [92.1, 96.2] | -0.1 [-0.4, +0.3] | 19.2 / 43.3 |
| v16d4e4 @1.0 (standard 4-bit) | 86.5% [82.3, 90.0] | -1.8 [-5.1, +0.8] | 92.9% [90.0, 95.2] | -1.5 [-3.7, -0.1] | 17.1 / 59.6 |
| v8d4e4 @1.0 | not read | | 93.0% [90.0, 95.4] | -1.4 [-3.6, -0.1] | - / 39.0 |
| v8d4e4 @0.7 | 88.4% [84.1, 91.9] | +0.1 [-1.5, +1.8] | 60.3% [52.7, 67.7] | **-34.1 [-42.1, -26.6]** | 9.2 / 22.2 |

- **`v8d8e8` matches the original on pages it was not found on**, on both sets, with narrow intervals. This is the confirmation `held/` could not give. It is better than the standard 4-bit build on `chatgpt` (+1.5, CI +0.1 to +3.7); on `spark` the difference has the same sign and is not shown (+1.8, CI -0.6 to +4.9).
- **A fixed 0.7x scale fails on dense small print.** The `chatgpt` pages hold about 4000 characters each, with small print 14 pixels high. At 0.7x `v8d4e4` loses 34 points and gets worse on every one of the 24 pages; at full resolution the same weights lose 1.4. The output is as long as the original's, with single characters wrong inside amounts and codes: `33.966.582,29 €` comes back as `33.966,582,29 €`. On the airier `spark` pages 0.7x costs nothing. So the "small and fast" configuration of the summary is safe only where the print is not small, and nothing in `hard/` or `held/` could show it: their smallest print is larger.
- **A 4-bit decoder costs about 1.5 points at full resolution** on `chatgpt`, encoder at 8 bits or in bf16 alike, and one repetition loop each.
- **The original model drops footers** on `spark`: the last three lines of the page are all missing on 14 pages of 60, at every level, for every configuration. The levels of `spark` do not order its pages by difficulty (88% at level 1, 85% at level 5).
- No Italian amount came back in English notation: 0 of 394 on `spark`, 0 of 834 on `chatgpt`.
- On `chatgpt` the count of accented words is decided by one word: the original model and `v8d8e8` write the table header `unità` as `uniti` on about half of the pages (168 misses, all that word), the 4-bit variants do not. It says nothing general about accents.

## Results on the tuning set

`hard/`, 40 pages, all at 0.7x. These are the runs the configuration was chosen on; the held-out table above is the one to trust.

| Config | Size | Entity recall [95% CI] | L1 | L2 | L3 | L4 | L5 | s/page | gen tok/s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 2651 MB | 81.6% [72.1, 90.1] | 100 | 100 | 94 | 86 | 30 | 10.4 | 67 | 3.07 |
| **v8d4e4** | 848 MB | 81.0% [71.2, 89.8] | 100 | 100 | 92 | 86 | 29 | 7.0 | 135 | 1.74 |
| v16d4e4 | 1247 MB | 81.0% [71.1, 89.7] | 100 | 100 | 92 | 87 | 28 | 6.9 | 132 | 2.13 |
| v16d8e8 | 1584 MB | 81.4% [71.9, 90.1] | 100 | 100 | 94 | 85 | 30 | 8.6 | 91 | 2.44 |
| v8d8e8 | 1184 MB | 80.7% [70.6, 89.6] | 100 | 100 | 90 | 87 | 29 | 8.2 | 88 | 1.99 |

- At 0.7x no variant differs from bf16 by more than 0.8 points, and every paired interval contains zero.
- A 4-bit decoder generates twice as fast as bf16 (135 against 67 tokens/s). Reading the image (about 450 tokens/s) costs the same for every variant, which is why cutting visual tokens is what saves time.
- The scale sweep on this set looked non-monotonic (77.2% at 0.85x against 81.0% at 0.7x). On the held-out set that dip is gone (+0.9): three pages had produced it.

### Lighter variants, first attempts

Same 40 pages, 0.7x, against `v8d4e4`. `g128` = quantization groups of 128 values instead of 64 (each group stores its own scale and offset, 69 MB of the 848).

| Config | Size | Difference from v8d4e4 [95% CI] | Pages with a skipped block |
|---|---|---|---|
| v8d4e4 | 848 MB | | 1 |
| v8d4e4g128 | 813 MB | +1.0 [-0.2, +2.6] | 0 |
| v6d4e4g128 | 707 MB | +0.8 [-0.4, +2.3] | 0 |
| v6d4e4 | 741 MB | -3.3 [-10.8, +1.2] | 1 |
| v5d4e4 | 688 MB | -0.4 [-1.8, +1.0] | 1 |
| v4d4e4 | 634 MB | -3.9 [-7.8, -1.0] | 1 |

- **An encoder at 4 bits is too few**: the loss is real and spread over 12 pages.
- **Larger groups cost nothing here**, and a 6-bit encoder with groups of 128 is 141 MB lighter than `v8d4e4` with no measurable loss.
- **This is not a decision.** Whether a block gets skipped changes from one variant to the next on a single invoice page (the two 6-bit rows differ by 4 points for that reason alone). Forty pages cannot rank these; the larger tuning set of `docs/ROADMAP.md` will.

Earlier exploration, in `archive/results/` and not in the tables: an encoder at 3 bits collapses into repetition loops on every page; the two LM Studio MLX builds (`lms4`, `lms8`) keep the encoder in bf16.

## Reproducibility

Checked again on 2026-10-13, after the repository was reorganised into the `glmocr` package:

- `pytest` recomputes every number of the tables above from `results/` (recall, intervals, levels, skipped blocks, loops, timings, paired differences) and fails if one differs.
- `generators/make_hard.py` regenerates the 160 pages of `hard/` and `held/` pixel for pixel (same SHA-256 for every image before and after the reorganisation), with identical ground truth and metadata.
- The source weights match the upstream checksum (`glmocr prepare` pins revision and SHA-256).
- `glmocr quant v8d4e4`, run again from a fresh download of the source checkpoint, gives a file with the same SHA-256 as the one measured (`cec967e1...e7ee23`).
- **Outputs are identical byte for byte only within one environment.** In today's fresh environment the original `run.py` and the new package give the same text on 40 pages of `hard/` out of 40. Against the saved run (`C_v8d4e4`), with the same weights, the same images and the same pinned packages, only 19 of 40 are identical: the others differ by a few characters, and five L5 pages by more. Entity recall moves from 81.0% to 81.4% (paired +0.4, CI -1.9 to +2.6; `rebuild_v8d4e4_0.7`). The previous environment is gone, so the cause is not established: the packages that were not pinned then, or the system. Consequences: two runs are compared only if they come from the same environment; the baselines have to be read again before any new comparison; `requirements.lock` now pins every package, and each new result file records its environment.
- **The four baselines read again in the new environment** (`E2_*`, `hard/` at 0.7x) against the tuning table above:

  | Config | Old | New | New - old [95% CI] | Pages identical | Skipped, old / new |
  |---|---|---|---|---|---|
  | bf16 | 81.6% | 82.2% | +0.6 [-0.7, +2.2] | 20/40 | 0 / 0 |
  | v8d4e4 | 81.0% | 81.4% | +0.4 [-1.9, +2.6] | 19/40 | 1 / 2 |
  | v16d4e4 | 81.0% | 81.6% | +0.6 [-1.8, +2.8] | 17/40 | 1 / 2 |
  | v8d8e8 | 80.7% | 82.2% | +1.4 [-0.6, +4.7] | 20/40 | 0 / 0 |

  Every configuration moves, by less than its interval, and what the table says does not change: within the new environment no variant differs measurably from bf16 (v8d4e4 -0.8, CI -3.0 to +0.9; v16d4e4 -0.6, CI -2.9 to +1.3; v8d8e8 0.0, CI -0.6 to +0.6), and the 8-bit decoder still skips no block. Peak memory and timings are the same as before. The memory fences do not change a single output (40 pages of 40 identical with and without them).
- Decoding is greedy and deterministic: re-running bf16 and v8d4e4 at 0.7x on `hard/` two days and one reboot after the first runs gave output identical byte for byte on 40 pages out of 40 (`V_*` against `C_*`, checked by the tests).
- `glmocr quant v16d4e4` reproduces the published `mlx-community/GLM-OCR-4bit` bit for bit (checked on 2026-10-12, not repeated since).

Limits: page images are not in the repository. `make_hard.py` draws them with macOS system fonts, so they regenerate on a Mac only. Timings come from single passes on one machine; treat them as good to about 10%.

## Failure modes to guard against

- **Repetition loops.** On the worst pages the decoder repeats one line until it runs out of tokens: 3 of 120 held-out pages for v8d4e4 at 0.7x, 4 for the original model, all on L4 and L5. It is not an effect of compression. `glmocr infer` detects it (token budget exhausted, or text that compresses below 25% with zlib) and exits with code 2. On the synthetic sets, no saved output of 200 bytes or more compresses to between 21% and 27%, and the threshold sits in that gap. On real pages it does not: right outputs of the Gazzetta Ufficiale, XFUND, DocAtlas, OmniDocBench and of sets whose sentences repeat by design also fall there (73 by 2026-10-15). There the ratio flags pages that did not loop; the retry it starts costs time, not quality, since the better reading is kept, and a loop is told by the token limit. Since 2026-10-14 the reader also stops a repetition as it starts (`metrics.loop_start`), after a fifth of its length in the median, and still counts the page as a loop.
- **Skipped blocks.** They come from the 4-bit decoder: `v8d8e8` does not have them. For `v8d4e4` no cheap detector was found. Two estimates of how much text a page holds, from connected components and from line geometry, were tried on `hard/` and are far too noisy (ratio to the true length from 0.2 to 1.7). Reading each page at two scales and keeping the longer output gains 0.7 points for three times the cost, and even picking the better output with the ground truth in hand gains only 2.3 (`glmocr adaptive`). If skipped small print matters more than speed, use `v8d8e8` at full resolution.
- **Rotated pages.** A page rotated by 90 degrees is unreadable for every variant. Straighten it first: the pretrained `PP-LCNet_x1_0_doc_ori` classifier (PaddleX, 30 ms/page on CPU) is right on 159/160 rotations of `hard/` and 475/480 of `held/`, with every error on L5 (`experiments/orient_make.py`, `orient_eval.py`). End to end on `held/`, straightening the 12 rotated pages with the classifier's own answer (10 right, 2 wrong) raises v8d4e4 at 0.7x from 80.1% to 82.1%, +2.0 points with a CI of +0.8 to +3.5 (`experiments/orient_fix.py`); on those pages the entities found go from 18 to 46 of 128, because they are all L5. The original model gains 1.1 points the same way (83.3% to 84.4%).

## Not done

- **Real Italian invoices and photographs of documents.** Real Italian pages measured are forms (XFUND), 40 documents of DocAtlas and pages of the Gazzetta Ufficiale (dev); invoices and phone photos are not. SROIE (real receipts, English) was read on a subset of 100.
- **Comb fields, blanks and tick boxes in training.** Three fine-tuning runs (op2 to op4, `docs/ROADMAP.md`) were worse than v1.1.0 on XFUND Italian train once both are read with the retry; the cause is in what op2 adds to the release's data, and is not isolated yet. The release avoids comb-field loops at reading time only.
- **Speed.** The release is as fast as the original. Fewer visual tokens per page is where time is saved (0.7x halves it), but it loses small print; reading a page in pieces (`docs/ROADMAP.md`) is a separate line of work, not started.
- **Smaller than 1184 MB with fine-tuning.** The 4-bit decoder skips small print; widths between 4 and 8 bits were not measured.
- **Vocabulary pruning** to Italian and English: tried, about 70 MB to gain, abandoned because the renumbered tokenizer did not load.
- **GGUF / ONNX packaging.**
