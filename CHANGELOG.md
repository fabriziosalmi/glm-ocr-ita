# Changelog

Versions of the released model and of the `glmocr` package go together. Dates are those of the measurements.

## Unreleased

- **Region mode.** `glmocr infer --regions` reads a page after GLM-OCR's own pipeline: its SDK's layout detector and formatter (Z.ai, Apache-2.0, unmodified, in their own PyTorch environment: `GLMOCR_LAYOUT_PYTHON` or `--layout-python`), each region read by the MLX reader with the prompt of its kind. Headers, footers, page numbers and footnotes are read unless `--drop-margins`. Not the default: worse on real Italian forms (docs/RESULTS.md, "Since v1.2.0").
- **Measured.** OmniDocBench `omnidoc_test2` by regions: overall (formulas by edit distance) 93.7 against 64.1 read whole; v1.1.0 not worse than the original on English or Chinese text. ocr-bench-v1 test half: +8.1 [+4.2, +13.6] over the original.
- **New.** `generators/make_ocrbench.py`; `experiments/omnidoc_compare.py` takes the set, its languages, formulas, tables and page types; `glmocr score` reads runs made by regions.
- **Training answers checked against the teacher's reading** (`experiments/ft_targets.py`, checks in `glmocr/metrics.py`): a page is left out when its ground truth lacks what the teacher reads, holds fewer tick boxes or blanks, is in another order, or flattens a form into one line. `--max-missing 2 --min-order 0` gives back the answers of op1 and op2 byte for byte.
- **DocAtlas training pages** without markdown escapes (`make_docatlas.py --train`); the test sets are unchanged.

## v1.2.0 - 2026-10-15

Same weights as v1.0.0 and v1.1.0 (same SHA-256); PDFs are read differently.

- **PDF text layer.** `glmocr infer file.pdf` reads each page from its own text layer where it passes four checks, and with the model only where it does not (`glmocr/pdf.py`; `--mode auto|ocr|text`, `--pages`). The idea and the checks come from Indra OCR by Datalab ([github.com/datalab-to/indra](https://github.com/datalab-to/indra), `indra_ocr/textlayer.py`, Apache-2.0; `NOTICE`), adapted. Two changes are ours: Latin text whose font map comes out as another script (the Gazzetta Ufficiale's small caps) is not trusted, and characters beyond the BMP are joined from pdfium's two UTF-16 halves before they are checked.
- **Measured** (docs/RESULTS.md, "Release v1.2.0"): on 80 real PDF pages of Omni Parse Bench (Datalab), scored by its own scorer, 76.1% of the text tests against 62.5% for v1.1.0, +12.2 [+5.8, +19.6]; +8.2 [+2.0, +16.3] on the tests not derived from the PDF. v1.1.0 against the original on the same pages: +6.9 [+2.2, +12.4]. Same weights, measured again: XFUND val German, Spanish, French, Portuguese all better than the original; 40 real Italian documents of DocAtlas not worse, clean or scanned.
- **New.** `experiments/pdf_eval.py` (a page set read from its PDFs), `experiments/opb_score.py` and `generators/make_opb.py` (a reduced Omni Parse Bench), `generators/make_docatlas.py`, `make_gu.py` (Gazzetta Ufficiale), `split_chatgpt.py`; `ft_targets.py` leaves out a page whose ground truth lacks what the teacher reads. Dependency added: `pypdfium2`.
- **Not released.** Fine-tuning runs op2 to op4 (comb fields; blanks, tick boxes and large amounts) are worse than v1.1.0 on XFUND Italian train with the retry (-2.1 to -2.3): `docs/ROADMAP.md`.
- **Corrected.** The loop threshold's "gap" holds on synthetic pages only: on real pages right outputs also compress to 21-27% (docs/RESULTS.md, failure modes).

## v1.1.0 - 2026-10-14

Same weights as v1.0.0 (same SHA-256); the reading changes.

- **Retry on loops.** A page that loops is read a second time with mlx-vlm's `repetition_penalty=1.1`; the second reading is kept unless it loops too and returns less text (`engine.keep_retry`). On by default in `glmocr infer` (`--retry-penalty 0` turns it off); off in `glmocr run` unless asked. Penalty chosen on XFUND Italian train (149 real forms, dev): 1.05, 1.1, 1.2 tried; 1.1 closes 5 of 6 loops.
- **Measured** (docs/RESULTS.md, "Release v1.1.0"): on XFUND Italian val (50 real forms, test) +3.1 [+1.3, +5.9] over the original, filled-in values +1.1 [-0.9, +3.2], loops 2 -> 0. v1.0.0 alone was +0.3 [-3.0, +2.9] there, with two new loops on comb fields. On an OmniDocBench subset (200 pages, official evaluator, text blocks) English is not shown either way (-0.3 [-2.7, +1.7]) and Chinese is worse (-4.8 [-10.6, -0.9]), with or without the retry.
- **New.** Converters for XFUND and an OmniDocBench subset, the official OmniDocBench evaluator wrapper (`experiments/omnidoc_eval.py`), the retry on saved runs (`experiments/loop_retry.py`), per-type comparison. Training pages for comb fields, not yet used: the official F24 filled with made-up values, twelve printable Italian forms, continuous-severity degradation.

## v1.0.0 - 2026-10-14

First release of weights: `GLM-OCR-ita-mlx-8bit` on Hugging Face (`fabriziosalmi/GLM-OCR-ita-mlx-8bit`).

- **Model.** `zai-org/GLM-OCR` (revision `2e85a62`) with LoRA adapter `op1` (rank 8, decoder only, 90 steps, learning rate 1e-5, 91 synthetic pages the original misread) fused into the bf16 weights, then quantized: encoder, decoder, embeddings and output head at 8 bits, groups of 64. 1184 MB, `model.safetensors` SHA-256 `f98e8f0c4e26903f71cccd2c115add8000a150141963da6f7b51ed0624d61ccd`.
- **Measured** (docs/RESULTS.md, "Release v1.0.0"): on the `spark/itdoc_b` test set Italian +3.7 [+2.4, +5.1] and English +7.1 [+4.1, +10.5] against the original; not worse on `chatgpt/itdoc_c` (+0.7 [+0.0, +1.4]) and on 100 real SROIE receipts (+0.0 [-1.2, +1.5]). Peak memory 2.10 GB against 3.16, 72 generated tokens/s against 65.
- **Package.** `glmocr infer` reads with the release by default, at full resolution. `glmocr quant` takes `--src` and `--out`; `experiments/ft_merge.py` fuses an adapter and checks the result against mlx-vlm's own fuse; `experiments/compare_lang.py` compares by language. `tests/test_release.py` recomputes the release table where the page sets are on the machine.
- **Fixed on the way.** Every read uses the official prompt (`Text Recognition:` with no `/nothink` and no empty think block); fine-tuning computes its loss with the causal mask that mlx-vlm 0.7.6 leaves out ([mlx-vlm#2509](https://github.com/Blaizzy/mlx-vlm/issues/2509)); the reader stops a repetition loop as it starts.

## v0.2.0 - 2026-10-13

No weights published. The `glmocr` package and command; compression study (`v8d8e8` = the original at 45% of the size, `v8d4e4` at 0.7x = small and fast); synthetic sets `hard/` and `held/`; roles of the commissioned page sets; memory fences; the test-set ledger.
