---
license: mit
base_model: zai-org/GLM-OCR
base_model_relation: finetune
library_name: gguf
pipeline_tag: image-to-text
language:
- it
- en
tags:
- ocr
- gguf
- llama.cpp
- italian
- 8-bit
---

# GLM-OCR-ita-GGUF

The weights of [GLM-OCR-ita-mlx-8bit](https://huggingface.co/fabriziosalmi/GLM-OCR-ita-mlx-8bit) ([GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) fine-tuned for Italian documents) in GGUF, for [llama.cpp](https://github.com/ggml-org/llama.cpp): Linux, Windows and macOS, CPU or GPU. A portable variant, measured on its own: it is not shown equivalent to the MLX release (Results). On Apple Silicon the MLX release remains the reference.

Code, page sets, raw outputs and the full measurement record: [github.com/fabriziosalmi/glm-ocr-ita](https://github.com/fabriziosalmi/glm-ocr-ita).

## Files

| File | Size | SHA-256 |
|---|---|---|
| `glm-ocr-ita-Q8_0.gguf` (language model, Q8_0) | 950 MB | `c8684098962b80bc8d8f61366a6dde20d803979da10ade196444fdf2203deda8` |
| `mmproj-glm-ocr-ita-f16.gguf` (vision encoder and projector, f16) | 869 MB | `3f92af9b6ad3078b831d0404e20dcb45a73ef4d6736a49bc21774bae40289369` |

Both are needed. Built with llama.cpp b11707 (`1268edac5`) from the fine-tuned bf16 checkpoint, the same one the MLX release was quantized from: `convert_hf_to_gguf.py` (text model in bf16, then `llama-quantize` Q8_0; the projector with `--mmproj` in f16). The multi-token-prediction layer is in the file and unused.

## Results

Entity recall: share of the entities of a page (amounts, dates, IBAN, VAT numbers, tax codes, CIG and CUP codes, PEC addresses) that come back exactly, whitespace aside; on XFUND, its annotated segments of 2 characters or more. Both models read the same pages on a Mac mini M4, greedy decoding, each page that loops read a second time with repetition penalty 1.1 (as the MLX release). Intervals are a 95% bootstrap over pages, differences paired. These are development sets: no test set was read with this model.

| Page set | Pages | MLX release | **This model** | Difference [95% CI] |
|---|---|---|---|---|
| Synthetic Italian and English pages (`spark/itdoc_a`) | 60 | 90.0% | **91.3%** | **+1.3 [+0.3, +2.5]**, better |
| of which Italian | 50 | 90.1% | 91.9% | +1.8 [+0.7, +3.0], better |
| of which English | 10 | 89.6% | 88.3% | -1.2 [-3.1, +0.9], not shown |
| Dense synthetic Italian documents (`chatgpt/itdoc_a`) | 24 | 94.7% | 94.5% | -0.2 [-1.6, +0.9], not shown |
| Gazzetta Ufficiale pages | 84 | 97.6% | 96.6% | -1.1 [-4.0, +1.0], not shown |
| XFUND real Italian forms (train split, never trained on) | 149 | 80.2% | 81.3% | +1.1 [-0.1, +2.4], not worse |
| Synthetic stress pages (`ocr-bench-v1`, train half) | 50 | 94.0% | 93.6% | -0.4 [-1.1, +0.4], not worse |

The project's rule calls a model equivalent when the lower bound is above -1.5 on every set: here it is not, on two sets and on English apart. Each of the two losses is one page. On a Gazzetta page both models fall into the same repetition loop; the MLX release's second reading comes out of it, this model's does not (llama.cpp applies the penalty, but at 1.1 it changes no token on that page). On a degraded synthetic page this model writes the thousands point as a comma in 12 amounts (38.638,964,49), the MLX release in 2.

| | Size | Memory | Seconds a page |
|---|---|---|---|
| MLX release, Mac mini M4 (Metal) | 1184 MB | 2.1 to 2.6 GB peak | 17 to 37 |
| **This model**, Mac mini M4 (Metal) | **1819 MB** | **2.6 to 2.8 GB resident** | **16 to 30** |
| **This model**, Docker on linux/amd64, 4 cores of an i7-6700 (CPU) | | 3.4 GB resident | about 260 |

On linux/amd64 (llama.cpp b11706 on the CPU) 10 synthetic Italian pages came back with the same entities as on the Mac mini, the text identical on 6 of them.

## Use

With llama.cpp:

```bash
llama-mtmd-cli -m glm-ocr-ita-Q8_0.gguf --mmproj mmproj-glm-ocr-ita-f16.gguf \
  --image page.png -p "Text Recognition:" --temp 0 -n 3000 -c 8192
```

With `llama-server` and its `/completion` endpoint, write the prompt in full, `[gMASK]<sop>` included: the server does not add them, and without them the page loops. The image marker is the server's `media_marker` from `/props`.

```text
[gMASK]<sop><|user|>
<media_marker>Text Recognition:<|assistant|>
```

If the text ends in a repetition loop, read the page again with `repeat_penalty` 1.1 and `repeat_last_n` 20, and keep that reading unless it loops too and is shorter.

With the `glmocr` harness, which starts and stops `llama-server`, stops loops, reads again with the penalty and guards memory (`llama-server` from llama.cpp on PATH, or `GLMOCR_LLAMA_SERVER`):

```bash
pip install git+https://github.com/fabriziosalmi/glm-ocr-ita
glmocr infer page.png --model fabriziosalmi/GLM-OCR-ita-GGUF   # the default off Apple Silicon
glmocr infer file.pdf --model fabriziosalmi/GLM-OCR-ita-GGUF   # each page from its text layer where it can be trusted
```

Or with Docker (linux/amd64 and linux/arm64), built from the repository:

```bash
docker build -t glmocr https://github.com/fabriziosalmi/glm-ocr-ita.git
docker run --rm -m 6g -v "$PWD/models:/models:ro" -v "$PWD:/data" glmocr infer page.png --model /models/glm-ocr-ita-Q8_0.gguf
```

## Limits

- **Not shown equivalent to the MLX release** (Results). Measured on development sets only.
- **Slow on a CPU**: about 260 seconds a page on 4 shared cores of an i7-6700. A GPU build of llama.cpp is what makes it practical.
- **Images over 4096 image tokens are resized by llama.cpp** for this projector (the MLX release reads up to 4968 on A4 pages at 200 dpi): on 50 such pages, -0.4 [-1.1, +0.4].
- **Repetition loops**, on comb fields and degraded pages, as for the MLX release; the second reading with the penalty rescues fewer of them here (5 of 9 against 7 of 11).
- **Amounts of tens of millions** on degraded pages are sometimes written with a comma as thousands separator.
- **Windows** is supported by the harness's code but not tested. Chinese and the other European languages are not measured for this model.
- Pages rotated by 90 degrees are unreadable, as for the original: straighten them first.

## License and citation

MIT, as the base model. Base model: GLM-OCR by Z.ai; cite the GLM-OCR technical report ([arXiv 2603.10910](https://arxiv.org/abs/2603.10910)). Converted and quantized with llama.cpp by ggml-org (MIT). XFUND: Xu et al., ACL Findings 2022 (CC BY-NC-SA 4.0), used for evaluation only.
