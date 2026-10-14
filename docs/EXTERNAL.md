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

**XFUND, Italian validation split** (downloaded on 2026-10-14 from the v1.0 release of github.com/doc-analysis/XFUND; `it.val.json` SHA-256 `3a3d505790607b7cf3e2c581c29ba317abec9a4059f04f378cbb8c3fba47bb7a`, `it.val.zip` `4e2232d9abb562bbd2b394cf8a2932ceda20f7402eeb61a7ef65859bbead9d16`): 50 real Italian forms (residence declarations, applications, administrative forms), scanned or photographed, filled in, with every text segment transcribed and labelled by hand (question, answer, header, other). License CC BY-NC-SA 4.0 (Xu et al., ACL 2022 Findings): used for evaluation only; forms, transcriptions and per-form outputs stay in `external/` and in git-ignored `results/xfund*`, and only aggregate numbers are published, with a citation. `generators/make_xfund.py` converts it. It is the first set of real Italian documents in this work, and a `test` set from the start. The Italian training split (149 forms; `it.train.json` SHA-256 `a5513300f86719b458b429f036ed8c4f6a8f714dbebdf2e4d6af586466ba0a81`, `it.train.zip` `8a721084c1d265974fbfc8f32d72fec15ed84db4799b3efbd0a95c4380fed6dc`) is a `dev` set, for choosing settings; it is never trained on, so that the weights carry no CC BY-NC-SA material.

Files in `external/` are untrusted: nothing in them is executed, and the bundled model files are never loaded.

**OmniDocBench v1.5** (downloaded on 2026-10-14 from `opendatalab/OmniDocBench` on Hugging Face, revision `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`; `OmniDocBench.json` SHA-256 `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`; evaluator github.com/opendatalab/OmniDocBench at commit `f133a71`, Apache-2.0): 1651 real pages, Chinese and English, from books, papers, slides, exams, magazines, newspapers. `generators/make_omnidoc.py` draws a fixed subset of 150 English and 50 Chinese pages, in proportion to the data sources. It is the benchmark GLM-OCR reports 94.6 on, with its own layout pipeline; we read whole pages with `Text Recognition:`, so our numbers are a paired comparison of the original and the release with the official evaluator, not a leaderboard score. In the evaluator's environment `pyarrow` is pinned to 14.0.2 and `datasets` to 2.19.2 (with `fsspec` 2024.3.1): its own pins (numpy 1.24.4) do not import otherwise.
