---
license: mit
base_model: zai-org/GLM-OCR
base_model_relation: finetune
library_name: mlx
pipeline_tag: image-to-text
language:
- it
- en
tags:
- ocr
- mlx
- mlx-vlm
- apple-silicon
- italian
- 8-bit
---

# GLM-OCR-ita-mlx-8bit (v1.0.0)

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), fine-tuned for Italian documents and quantized to 8 bits for Apple Silicon with [MLX](https://github.com/ml-explore/mlx). It reads Italian better than the original, and English too, at 45% of the size.

Code, test sets, raw outputs and the full measurement record: [github.com/fabriziosalmi/glm-ocr-ita](https://github.com/fabriziosalmi/glm-ocr-ita).

## Results

Entity recall: share of the entities of a page (amounts, dates, IBAN, VAT numbers, tax codes, CIG and CUP codes, PEC addresses, contract numbers) that come back exactly, whitespace aside. Intervals are a 95% bootstrap over pages; differences are paired on the same pages. M4 with 16 GB, greedy decoding.

| Page set | Pages | Original (bf16) | **This model** | Difference [95% CI] |
|---|---|---|---|---|
| Synthetic Italian pages, renderer never trained on (`spark/itdoc_b`) | 95 | 88.7% | **92.4%** | **+3.7 [+2.4, +5.1]** |
| Synthetic English pages, same renderer | 25 | 86.8% | **93.9%** | **+7.1 [+4.1, +10.5]** |
| Dense synthetic Italian documents (`chatgpt/itdoc_c`) | 24 | 93.0% | 93.7% | +0.7 [+0.0, +1.4] |
| SROIE 2019 real scanned receipts, English (subset) | 100 | 92.0% | 92.0% | +0.0 [-1.2, +1.5] |

| | Size | Peak memory | Generation speed |
|---|---|---|---|
| Original (bf16) | 2651 MB | 3.16 GB | 65 tokens/s |
| **This model** | **1184 MB** | **2.10 GB** | **72 tokens/s** |

Most of the gain is text the original leaves out: letterheads and footers (pages returning under 90% of their text drop from 25 to 3 of 120). Tax codes go from 60% to 72%, IBANs from 72% to 84%, VAT numbers from 73% to 87%.

Every test set was read once for this model, with its question and threshold written down before the read.

## Limits

- **Every Italian page measured is synthetic**, from three page generators commissioned for this work. Real Italian documents have not been measured.
- **Not faster than the original per page**: generation is faster, reading the image costs the same.
- Repetition loops are rare but happen on degraded pages (2 of 120 on the test set, on pages where no entity was lost). Stop generation when the output starts repeating itself; the `glmocr` package does it.
- Pages rotated by 90 degrees are unreadable, as for the original: straighten them first.

## Use

The prompt must be the official one: `Text Recognition:` after the image, and nothing in the assistant turn. mlx-vlm's chat template adds `/nothink` and an empty think block, which this model was not trained with: remove them.

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("fabriziosalmi/GLM-OCR-ita-mlx-8bit")
prompt = apply_chat_template(processor, model.config, "Text Recognition:", num_images=1)
prompt = prompt.replace("/nothink", "").replace("<think></think>\n", "")   # the official prompt
print(generate(model, processor, prompt, image="page.png", max_tokens=3000, temperature=0.0).text)
```

Or with the harness, which also guards memory and stops loops: `pip install -r requirements.lock -e .` in the repository, then `glmocr infer page.png`.

Tested with mlx 0.32.3 and mlx-vlm 0.7.6 (`requirements.lock` in the repository). `model.safetensors` SHA-256: `f98e8f0c4e26903f71cccd2c115add8000a150141963da6f7b51ed0624d61ccd`.

## How it was made

1. **Data.** 152 synthetic pages from 19 page sets of one generator (`gemini`), Italian and English, levels 1 to 3 (clean to poor scan). The original model read each; the 91 pages where it missed an entity or returned under 90% of the words were kept. Target = the ground-truth text.
2. **Training.** LoRA, rank 8, alpha 16, on every linear layer of the decoder; vision encoder and projector frozen. 90 steps of one page, learning rate 1e-5, Adam, 27 minutes on an M4. Loss on the answer only, with the causal mask (mlx-vlm 0.7.6 omits it when training this model: [mlx-vlm#2509](https://github.com/Blaizzy/mlx-vlm/issues/2509)), answer formatted exactly as the model writes it at inference and closed by the token it stops on.
3. **Selection.** Adapters at steps 30 and 90 judged on a fixed panel against a stopping rule written before training (English not more than 1.5 points worse; no new loop that costs entities; no more incomplete pages). Step 90 then validated on 84 pages of two other generators.
4. **Release.** Adapter fused into the bf16 weights (checked against mlx-vlm's own fuse, largest difference 1e-3), then quantized: every layer at 8 bits, groups of 64. The adapter alone is in `adapter/`, for use on the original bf16 weights.

## License and citation

MIT, as the base model. Base model: GLM-OCR by Z.ai; cite the GLM-OCR technical report ([arXiv 2603.10910](https://arxiv.org/abs/2603.10910)). SROIE: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction, used for evaluation only.
