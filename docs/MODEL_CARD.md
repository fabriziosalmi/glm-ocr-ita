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

# GLM-OCR-ita-mlx-8bit (v1.2.0)

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), fine-tuned for Italian documents and quantized to 8 bits for Apple Silicon with [MLX](https://github.com/ml-explore/mlx). It reads Italian better than the original, on real forms and on synthetic documents, and real forms in German, Spanish, French and Portuguese too, at 45% of the size. It is worse on Chinese: for Chinese documents, use the original.

Code, test sets, raw outputs and the full measurement record: [github.com/fabriziosalmi/glm-ocr-ita](https://github.com/fabriziosalmi/glm-ocr-ita).

## Results

Entity recall: share of the entities of a page (amounts, dates, IBAN, VAT numbers, tax codes, CIG and CUP codes, PEC addresses, contract numbers) that come back exactly, whitespace aside. Intervals are a 95% bootstrap over pages; differences are paired on the same pages. M4 with 16 GB, greedy decoding.

| Page set | Pages | Original (bf16) | **This model** | Difference [95% CI] |
|---|---|---|---|---|
| Synthetic Italian pages, renderer never trained on (`spark/itdoc_b`) | 95 | 88.7% | **92.4%** | **+3.7 [+2.4, +5.1]** |
| Synthetic English pages, same renderer | 25 | 86.8% | **93.9%** | **+7.1 [+4.1, +10.5]** |
| Dense synthetic Italian documents (`chatgpt/itdoc_c`) | 24 | 93.0% | 93.7% | +0.7 [+0.0, +1.4] |
| SROIE 2019 real scanned receipts, English (subset) | 100 | 92.0% | 92.0% | +0.0 [-1.2, +1.5] |
| **XFUND real Italian forms** (val), read with the retry below | 50 | 74.6% | **77.7%** | **+3.1 [+1.3, +5.9]** |
| XFUND real forms, German / Spanish / French / Portuguese (val) | 50 each | 84.8 / 81.2 / 73.2 / 81.1% | 87.3 / 85.0 / 77.2 / 84.3% | +2.5 / +3.8 / +4.0 / +3.2, all better |
| DocAtlas, 40 real Italian documents, clean / scanned | 40 | 98.4 / 98.4% | 98.3 / 98.1% | -0.1 / -0.3, not worse |

[Omni Parse Bench](https://github.com/datalab-to/omni_parse_bench) (Datalab), 80 real PDF pages from ten suites, its own scorer, 184 text tests: original 56.0%, this model 62.5%, **+6.9 [+2.2, +12.4]**, no page worse; English pages alone +4.0 [+0.0, +9.9]. Pages read whole at 150 dpi: not comparable with the benchmark's own table, which reads GLM-OCR region by region after layout detection.

OmniDocBench v1.5, a fixed subset of 200 pages, official evaluator, text blocks only (normalized edit distance x100, lower is better; formulas and tables not scored):

| Pages | Original (bf16) | This model (v1.1.0) | Original - this model [95% CI] |
|---|---|---|---|
| English, 140 with matched text | 7.6 | 7.9 | -0.3 [-2.7, +1.7], not shown either way |
| Chinese, 45 with matched text | 14.8 | 19.7 | **-4.8 [-10.6, -0.9], worse** |

| | Size | Peak memory | Generation speed |
|---|---|---|---|
| Original (bf16) | 2651 MB | 3.16 GB | 65 tokens/s |
| **This model** | **1184 MB** | **2.10 GB** | **72 tokens/s** |

Most of the gain is text the original leaves out: letterheads and footers (pages returning under 90% of their text drop from 25 to 3 of 120). Tax codes go from 60% to 72%, IBANs from 72% to 84%, VAT numbers from 73% to 87%.

Every test read had its question and threshold written down before the read ([LEDGER.md](https://github.com/fabriziosalmi/glm-ocr-ita/blob/main/docs/LEDGER.md)). XFUND val was read twice: once for v1.0.0, where the model was not shown better (+0.3 [-3.0, +2.9]) because it looped on 2 forms, then for v1.1.0, with the retry chosen on the XFUND training split (149 forms, never trained on: +2.9 [+1.1, +4.2] there).

## What changed in v1.2.0

The weights are those of v1.0.0 (same SHA-256). The `glmocr` harness now reads PDFs: each page from its own text layer where the layer passes four checks (not an invisible OCR layer, no broken font map, ink under its text, content order a reading order), and with this model only where it does not. The idea and the checks come from [Indra OCR](https://github.com/datalab-to/indra) by Datalab (`indra_ocr/textlayer.py`, Apache-2.0), adapted. On the Omni Parse Bench subset above this reads 76.1% of the text tests against 62.5% for the model alone, +12.2 [+5.8, +19.6] (+8.2 [+2.0, +16.3] on the tests not derived from the PDF itself); 29 of the 80 pages pass the checks, none of them read worse. With mlx-vlm alone there is no PDF mode: render the page and read the image.

## What changed in v1.1.0

The weights are those of v1.0.0 (same SHA-256). On forms with comb fields (one character per box, as in tax codes) the model can fall into a repetition loop and lose the rest of the page. v1.1.0 reads such a page a second time with `repetition_penalty=1.1` and keeps the second reading unless it loops too and returns less text. On XFUND val this closes every loop (original 2, v1.0.0 3, v1.1.0 0) at about 1 second per form on average. The `glmocr` harness does it by default; with mlx-vlm alone, see Use.

## Limits

- **Not faster than the original per page**: generation is faster, reading the image costs the same.
- **Chinese is worse** than with the original (OmniDocBench, above): no training page was Chinese.
- **Real Italian documents measured**: forms (XFUND, 50 test and 149 dev), 40 documents of DocAtlas (clean and scanned), pages of the Gazzetta Ufficiale (dev only). Invoices and phone photos of documents are not measured.
- **Amounts of tens of millions** on dense synthetic pages are sometimes written with a comma as thousands separator ("29.467,745,94"): worse than the original on 10 such pages, -5.3 [-11.3, -0.7]; on the Gazzetta's real amounts it reads more of them right than the original.
- Repetition loops happen on comb fields and degraded pages. Stop generation when the output starts repeating itself and read the page again with `repetition_penalty=1.1`; the `glmocr` package does both. One XFUND dev form still loops at every penalty tried.
- Pages rotated by 90 degrees are unreadable, as for the original: straighten them first.

## Use

The prompt must be the official one: `Text Recognition:` after the image, and nothing in the assistant turn. mlx-vlm's chat template adds `/nothink` and an empty think block, which this model was not trained with: remove them.

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("fabriziosalmi/GLM-OCR-ita-mlx-8bit")
prompt = apply_chat_template(processor, model.config, "Text Recognition:", num_images=1)
prompt = prompt.replace("/nothink", "").replace("<think></think>\n", "")   # the official prompt
text = generate(model, processor, prompt, image="page.png", max_tokens=3000, temperature=0.0).text
# if the text ends in a repetition loop, read again with repetition_penalty=1.1 (v1.1.0) and keep that reading
```

Or with the harness, which also guards memory and stops loops: `pip install -r requirements.lock -e .` in the repository, then `glmocr infer page.png` (the retry is on by default), or `glmocr infer file.pdf` (each page from its text layer where it can be trusted, else with the model).

Tested with mlx 0.32.3 and mlx-vlm 0.7.6 (`requirements.lock` in the repository). `model.safetensors` SHA-256: `f98e8f0c4e26903f71cccd2c115add8000a150141963da6f7b51ed0624d61ccd`.

## How it was made

1. **Data.** 152 synthetic pages from 19 page sets of one generator (`gemini`), Italian and English, levels 1 to 3 (clean to poor scan). The original model read each; the 91 pages where it missed an entity or returned under 90% of the words were kept. Target = the ground-truth text.
2. **Training.** LoRA, rank 8, alpha 16, on every linear layer of the decoder; vision encoder and projector frozen. 90 steps of one page, learning rate 1e-5, Adam, 27 minutes on an M4. Loss on the answer only, with the causal mask (mlx-vlm 0.7.6 omits it when training this model: [mlx-vlm#2509](https://github.com/Blaizzy/mlx-vlm/issues/2509)), answer formatted exactly as the model writes it at inference and closed by the token it stops on.
3. **Selection.** Adapters at steps 30 and 90 judged on a fixed panel against a stopping rule written before training (English not more than 1.5 points worse; no new loop that costs entities; no more incomplete pages). Step 90 then validated on 84 pages of two other generators.
4. **Release.** Adapter fused into the bf16 weights (checked against mlx-vlm's own fuse, largest difference 1e-3), then quantized: every layer at 8 bits, groups of 64. The adapter alone is in `adapter/`, for use on the original bf16 weights.

## License and citation

MIT, as the base model. Base model: GLM-OCR by Z.ai; cite the GLM-OCR technical report ([arXiv 2603.10910](https://arxiv.org/abs/2603.10910)). SROIE: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction; XFUND: Xu et al., ACL Findings 2022 (CC BY-NC-SA 4.0); OmniDocBench: Ouyang et al., CVPR 2025 (Apache-2.0); Omni Parse Bench: Datalab, 2026 (scorer Apache-2.0, tests CC BY 4.0, pages under the license of each source); DocAtlas (CC BY 4.0). All used for evaluation only. The PDF text-layer checks of the harness are adapted from Indra OCR by Datalab (Apache-2.0).
