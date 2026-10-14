# Changelog

Versions of the released model and of the `glmocr` package go together. Dates are those of the measurements.

## v1.0.0 - 2026-10-14

First release of weights: `GLM-OCR-ita-mlx-8bit` on Hugging Face (`fabriziosalmi/GLM-OCR-ita-mlx-8bit`).

- **Model.** `zai-org/GLM-OCR` (revision `2e85a62`) with LoRA adapter `op1` (rank 8, decoder only, 90 steps, learning rate 1e-5, 91 synthetic pages the original misread) fused into the bf16 weights, then quantized: encoder, decoder, embeddings and output head at 8 bits, groups of 64. 1184 MB, `model.safetensors` SHA-256 `f98e8f0c4e26903f71cccd2c115add8000a150141963da6f7b51ed0624d61ccd`.
- **Measured** (README, "Release v1.0.0"): on the `spark/itdoc_b` test set Italian +3.7 [+2.4, +5.1] and English +7.1 [+4.1, +10.5] against the original; not worse on `chatgpt/itdoc_c` (+0.7 [+0.0, +1.4]) and on 100 real SROIE receipts (+0.0 [-1.2, +1.5]). Peak memory 2.10 GB against 3.16, 72 generated tokens/s against 65.
- **Package.** `glmocr infer` reads with the release by default, at full resolution. `glmocr quant` takes `--src` and `--out`; `experiments/ft_merge.py` fuses an adapter and checks the result against mlx-vlm's own fuse; `experiments/compare_lang.py` compares by language. `tests/test_release.py` recomputes the release table where the page sets are on the machine.
- **Fixed on the way.** Every read uses the official prompt (`Text Recognition:` with no `/nothink` and no empty think block); fine-tuning computes its loss with the causal mask that mlx-vlm 0.7.6 leaves out ([mlx-vlm#2509](https://github.com/Blaizzy/mlx-vlm/issues/2509)); the reader stops a repetition loop as it starts.

## v0.2.0 - 2026-10-13

No weights published. The `glmocr` package and command; compression study (`v8d8e8` = the original at 45% of the size, `v8d4e4` at 0.7x = small and fast); synthetic sets `hard/` and `held/`; roles of the commissioned page sets; memory fences; the test-set ledger.
