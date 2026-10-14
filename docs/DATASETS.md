# Datasets

State on 2026-10-13. The page sets live outside this repository, under the datasets root (`$GLMOCR_DATASETS`, default `~/Documents/git/datasets`), one folder per renderer. Their roles are in `datasets.toml`; this file says why. `glmocr check` validates every set against the contract the scorer relies on.

The roles below were fixed before any model read any of these pages.

**The `gemini` sets were regenerated on 2026-10-13** and now pass `glmocr check`: real line breaks, 240 Italian and 60 English pages at every level, both languages in every template, every entity typed. Six sets arrived (`itdoc_a` to `itdoc_f`, 9000 pages), all training data. A 20-page smoke read of `itdoc_a` with the fast model: 84.5% entity recall, level 5 at 64%, no loop, no page over the token budget. A pilot read of the first version of `itdoc_a` had been discarded.

## Inventory on 2026-10-14: the generators have stopped, this is what there is

| Renderer | Sets | Pages | Balanced by language at every level | Role |
|---|---|---|---|---|
| `gemini` | 26 (`a` to `z`) | 39000 | yes | train |
| `spark` | 11 (`a` to `k`) | 16500 | yes | `a`, `c` to `g` dev; `h` to `k` test, never read; `b` test, read once with the old prompt |
| `chatgpt` | 26 (`a` to `z`) | 624 | no: the 4 English pages of each set are all level 5 | 25 dev, `c` test (read once, with the old prompt) |

All 63 sets pass `glmocr check`, apart from the language tie of `chatgpt` and 48 stray files in `chatgpt/itdoc_b`. Each `spark` set keeps the templates of the one before and adds four: 14 in `a`, 54 in `k`. The `chatgpt` sets are 24-page prototypes whose twelve templates share one layout; their pages are dense (about 77 entities each) with small print 14 pixels high, which is what they are good for. The table below is the first delivery and is kept for the record.

## Audit of the texts, 2026-10-14

`experiments/dataset_audit.py` (no model; texts and metadata only) wrote a manifest of 56211 files (the 63 `meta.json` and every `.txt`, SHA-256 each). The per-set digests are in `docs/datasets_digests.json`; the full manifest and the report are in `work/dataset_audit/`. **The 56148 images (76.7 GB) are not in the manifest yet**, and nothing below was measured on pixels. Lines marked (checked) were measured again on a sample outside the audit script.

Clean: no entity missing from its ground truth (of 996206), no identical texts, no escape sequences, and no content shared between `gemini` and the two evaluation renderers (no sentence pattern, company name, IBAN, tax code, CIG, CUP or PEC in common). No page risks the 3000-token budget.

Defects, most serious first:

1. **Ambiguous entities in `gemini`**: 22.8% occur more than once in their page, so they can be found by accident (checked: 20% on a sample; `spark` 0.007%, `chatgpt` 2.6%). Recall on `gemini` pages is inflated; the "inside" group of the panel is to be read with that in mind.
2. **`gemini` pages labelled English are largely Italian**: 56% have at least a fifth of Italian function words, all of them in sets `n` to `s` (checked: 65% on a sample). English is measured on `spark` only.
3. **`gemini` `t` to `z` are near-copies of themselves**: 24 to 36 distinct text skeletons per 1500 pages in `u` to `z`; 803 entity types outside the brief. Training pages are better drawn from `a` to `s`.
4. **Codes that are not valid**: `chatgpt` IBANs lack the Italian check letter (checked); `gemini` `p` to `t` have wrong IBAN check letters, tax codes and CUP formats. `spark` is valid throughout (checked for IBAN and VAT number). A check-digit test of OCR output can be evaluated on `spark` and on `gemini` `a` to `o`.
5. **`spark` writes Italian without accents** (`sara’`, `proprieta’`) on most Italian pages (checked). Accented-letter recall cannot be measured there.
6. **`spark` test sets are not independent of `spark` dev sets**: same templates, 68 to 87% of the sentence patterns of a test set exist in dev, and the 1500 PEC addresses are shared by the whole renderer. `h` to `k` add 4 to 16 templates that dev does not have. They are an independent test of a training on `gemini`; they are a weaker test of choices made on `spark` dev.
7. **Levels are not ordered** in the declared parameters of `spark` (level 5 has less blur than level 4) and `chatgpt` (levels 3 to 5 differ in JPEG quality only), which matches what the reads showed.
8. **In `spark`, VAT number, tax code and IBAN sit in the last three lines** two times out of three: recall of those types there is footer recall.

Corrections to what this file said before: `chatgpt` is 83/17 by language and its levels hold 130, 130, 130, 130 and 104 pages, not 80/20 in equal parts; "balanced by language" is true of the labels of `gemini`, not of its text; `gemini` has 270 distinct template names, not 312.

## What is there

Commissioned with `DATASET_PROMPT.md`, each renderer written by a different model, independently of `generators/make_hard.py` and of the others. All pages are 1240 x 1754, portrait, never rotated by 90 degrees; 80% Italian, 20% English; five degradation levels in equal parts.

| Set | Pages | Entities | Templates | Focus | Role | `glmocr check` |
|---|---|---|---|---|---|---|
| `gemini/itdoc_a` | 1500 | 21358 | 12 | invoices, delivery notes, purchase orders | **train** | passes |
| `gemini/itdoc_b` | 1500 | 19083 | 12 | contracts, agreements, annexes | **train** | passes |
| `spark/itdoc_a` | 1500 | 37689 | 14 | invoices, delivery notes, purchase orders | **dev** | passes |
| `spark/itdoc_b` | 1500 | 37153 | 18 | contracts, agreements, annexes | **test** | passes |
| `chatgpt/itdoc_a`, `_b`, `_d`, `_e` | 24 each | about 1900 each | 12 each | one focus each | **dev** | `_b` holds 48 files of an earlier generation that `meta.json` does not list; the 24 listed pages pass |
| `chatgpt/itdoc_c` | 24 | 1832 | 12 | public administration | **test** | passes |
| `gemini/itdoc_c` | 1500 | 19875 | 12 | public administration | **train** | passes |
| `gemini/itdoc_d`, `_e`, `_f` | 1500 each | 20636, 18929, 18650 | 12 each | see their README | **train** | pass |
| `gemini/itdoc_g` to `_z` | 1500 each, 30000 in all | about 20000 each | 12 each | one kind of Italian document per set (deeds, vehicle papers, risk assessments, residence permits, waste forms, ...) | **train** | pass; the first run does not use them |

Also there, with no role:

- `spark/itdoc_b24`: a 24-page preview of `spark/itdoc_b`. It is the test set's renderer and seed, so it is treated as test: do not read it.
- `huggingface/data/docatlas_it` and `docatlas_it_scan`: 2356 Italian pages from `ahmedheakl/docatlas_instruct` (CC-BY-4.0, documents rendered from DOCX, ground truth exact by construction) and the same pages with scan effects. They have a transcription but no entity list and are not in this repository's format, so the scorer cannot read them yet. They are the candidate for the "real public documents" third of the test set (Phase 0 of `ROADMAP.md`); their terms go in `EXTERNAL.md` before they enter a split.

## Why these roles

1. **`itdoc_c` was designated as test on 2026-10-12, before any set existed.** That stands for `chatgpt/itdoc_c`. The `gemini` rendering of it was moved to `train` on 2026-10-13, before any model read it, because a test set from the renderer of the training pages proves little.
2. **Evaluation is `spark` and `chatgpt`; fine-tuning is `gemini`.** A renderer is never on both sides of a training: no page of `spark` or `chatgpt` is ever trained on, and the three `gemini` sets are the only training data.
3. **`dev` and `test` share their renderers** (`spark` and `chatgpt` are on both sides) but not the focus or the seed. A configuration tuned on `dev` can still fit a renderer; a result counts when it holds on both renderers of `test`. Today the mix is lopsided: `spark` has 1500 pages per set, `chatgpt` 24.
4. **The `chatgpt` sets are previews**: 24 pages each, and the README of `itdoc_c` says its twelve "templates" are labels on one page construction. They are too few and too alike to decide anything alone.

## Accepting a set

Before a full set is generated: a 24-page preview, then `glmocr smoke NAME` (3 minutes). Before a full set is read: `glmocr smoke` again on the real thing. Hours of generation or reading start only after that.

## Known limits

- **In `spark/itdoc_a` the levels do not order the pages by difficulty.** On the 60-page pilot the original model finds 88% of the entities at level 1 and 85% at level 5, and the fast model the same. What this set measures is something else: the model drops the footer block on about a quarter of the pages (the last three lines all missing on 14 of 60) and sometimes the letterhead, at every level. Its level 5 is not a bad scan in the sense of `hard/`.

- **In the `chatgpt` sets, and in the first version of the `gemini` ones, language is tied to level.** Every English page is level 5 and every level 5 page is English; levels 1 to 4 are Italian only. On those sets "English recall" and "level 5 recall" are the same number, there is no Italian page at the worst level and no clean English page. They cannot serve rule 9 (English must not pay for Italian) and, as training data, would teach that bad scans are English. `spark` and the regenerated `gemini` are balanced (240 Italian and 60 English pages at every level). `glmocr check` reports it.

- **Short entities.** 2.5% (`spark`, `chatgpt`) to 7.6% (`gemini`) of the entities of these sets are under five characters (percentages such as `22%`), against none in `hard/` and `held/`. They can match by accident, more so on dense pages. Report recall by entity type and watch the `percent` column before trusting a small difference.
- **A full read is long.** 1500 pages at 15 to 25 s each is 6 to 10 hours per configuration. Phase 0 has to fix a subsample of `dev` and of `test` (by page index, before any read) sized so that the interval of BASE against NOW has a half-width of 1.5 points or less.
- **Levels 4 and 5.** On the first version of `gemini/itdoc_a` the small print of the two worst levels could not be read by a person either. The set has been regenerated since; this has not been looked at again.
- **`gemini/itdoc_a` is not the set of 2026-10-12.** It was regenerated during the night with other templates. The 18 pages behind the reading-in-pieces table of `ROADMAP.md` no longer exist, so that table cannot be recomputed; its raw outputs are still in `results/itdocval_*`.
