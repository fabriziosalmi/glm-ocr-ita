# External datasets

Downloaded on 2026-10-12 into `external/` (git-ignored). `external/` did not survive the fresh clone of 2026-10-12: the archive has to be downloaded again and checked against the SHA-256 below. The commissioned synthetic sets are not here: see `DATASETS.md`. Rule 8 of `ROADMAP.md` applies: a dataset enters a split only after its reuse terms are read and written here.

| Folder | Source | Size | License on Kaggle | SHA-256 of the archive | What is inside |
|---|---|---|---|---|---|
| `urbikn_sroie-v2` | kaggle.com/datasets/urbikn/sroie-datasetv2 | 875 MB | "Other (specified in description)" | `e4502ab8...e534844` | ICDAR 2019 SROIE: 973 scanned receipts in English (626 train, 347 test), each with the text of every line and four entities (company, date, address, total). Also a copy of `layoutlm-base-uncased`, not needed and never loaded |

**Terms.** The Kaggle page states no license: it is a regrouping of the data of the ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction (rrc.cvc.uab.es/?ch=13), released by the organizers for research. So here it is used for evaluation only. The receipts, their transcriptions and the per-receipt outputs of the models stay in `external/` and in `results/sroie_*`, both git-ignored; only aggregate numbers are published, with a citation of the challenge.

**Role.** The test split (347 receipts) is a frozen test set for the compression tracks, designated on 2026-10-12 before any model was run on it; reads are logged in `LEDGER.md`. `generators/make_sroie.py` converts it to this repository's format. The train split is free for tuning.

Downloaded the same day and discarded as not useful: `nadaarfaoui/ocr-processed-handwritten-prescriptions` (4680 crops of single handwritten medicine names, not pages) and `volkandl/optical-character-recognition-ocr-texts` (38 images with no ground truth).

Considered and not downloaded, for size: `osamahosamabdellatif/high-quality-invoice-images-for-ocr` (2.4 GB), `senju14/ocr-dataset-of-multi-type-documents` (5.7 GB).

SROIE is English. It serves the compression goal (real scans), not the Italian one.

Files in `external/` are untrusted: nothing in them is executed, and the bundled model files are never loaded.
