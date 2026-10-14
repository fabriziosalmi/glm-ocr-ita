# Archive

The first, smaller test bed of this project, kept because the loop threshold of `glmocr/metrics.py` was first set on its outputs. Nothing in the README tables depends on it, and nothing here is maintained.

- `corpus/` (27 pages): 8 synthetic pages plus pages rendered from the synthetic contracts of a separate project (`pdf-ocr-ng`), so `make_corpus.py` cannot be re-run from this repository alone.
- `eval/`, `make_eval.py`: the 8 synthetic pages.
- `score.py`: character error rate on `corpus/`.
- `results/`: runs on `corpus/` (`bf16`, `v3d4e4`, `v4*`, `v8*`), the two LM Studio MLX builds (`lms4`, `lms8`) and `U_bf16`.

The scripts expect to be run from this folder with the old flat layout; they are here as a record, not as tools.
