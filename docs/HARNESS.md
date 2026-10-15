# Harness

How to run, rebuild and measure: the `glmocr` package, its memory fences and the layout of the repository. Moved here from the README on 2026-10-15.

## Usage

Python 3.12 on Apple Silicon.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock -e ".[dev]"   # every package pinned
source .venv/bin/activate
glmocr infer page.png          # the release, downloaded from Hugging Face (1.2 GB) on first use; a page that loops is read again
                               # with repetition penalty 1.1 (--retry-penalty 0 to turn it off); exit code 2 if it still loops
glmocr infer file.pdf          # each page from its text layer where it passes the checks, else with the model; pages separated by
                               # a form feed; --mode ocr always uses the model, --pages 1-3,7 picks pages; -o out.json says which was which
```

With mlx-vlm alone, the prompt must be the official one: `Text Recognition:` right after the image, and nothing else in the assistant turn. mlx-vlm's own chat template adds `/nothink` and an empty think block, which the model was not trained with (`docs/MODEL_CARD.md` has the snippet).

To rebuild the release from the upstream checkpoint (the adapter is published next to the weights, in `adapter/`):

```bash
glmocr prepare                                                         # zai-org/GLM-OCR (2.65 GB) into work/glm-ocr-bf16, checksum pinned
python experiments/ft_merge.py work/ft/op1_90 work/ft/op1_90_merged    # adapter fused into the bf16 weights, checked against mlx-vlm's own fuse
glmocr quant v8d8e8 --src work/ft/op1_90_merged --out models/v8d8e8_op1
glmocr quant v8d8e8                                                    # the 8-bit compression without fine-tuning (1184 MB)
glmocr quant v8d4e4                                                    # small and fast (848 MB), for large print only: run it with --scale 0.7
```

To train the adapter again: `experiments/ft_data.py`, `ft_targets.py`, `ft_train.py` and `ft_points.py`, in that order; their docstrings give the commands of run `op1`, and `docs/ROADMAP.md` its settings and outcome. The training pages (`gemini`, `docs/DATASETS.md`) are not in this repository.

To redo the measurements:

```bash
pytest                                                                 # the published tables, from results/
python generators/make_hard.py                                         # hard/
python generators/make_hard.py --out held --seed 20261012 --reps 3 --variants   # held/
glmocr run models/v8d4e4 my_tag --scale 0.7 --corpus held              # -> results/my_tag.json, resumes if interrupted
glmocr score --corpus held my_tag                                      # recall, CI, skipped blocks, loops, by level and type
glmocr compare --corpus held held_bf16_1.0 my_tag                      # paired difference and its verdict
glmocr adaptive --corpus held held_v8d4e4_0.7 held_v8d4e4_1.0          # two-scale policies
glmocr check                                                           # validate every page set of datasets.toml
glmocr smoke spark/itdoc_a                                             # a new page set in 3 minutes: check, then 20 pages read
```

## Memory

Two model jobs together run out of memory on 16 GB, and the first one dies. `glmocr/guard.py` puts four fences around every job that loads a model (`quant`, `infer`, `run`, `experiments/crops.py`):

| Fence | What it does |
|---|---|
| One job at a time | An exclusive lock (`work/.model.lock`). A second job refuses to start and names the pid of the first |
| Room before loading | The weights plus 1 GB must fit under the ceiling, and the system must have that much available plus a floor of 1.5 GB, or the job does not start |
| A ceiling | 6 GB by default (`--mem-gb`, or `GLMOCR_MEM_GB`), set as MLX's memory limit; MLX's cache of freed buffers is capped at 1.5 GB |
| A watch per page | After every page the job stops, with exit code 3, if the page peaked over the ceiling or the system is under the floor. The page is saved first; the same command resumes the run |

MLX treats its limit as a guideline, so the watch is the fence that actually stops a job, and it acts between pages, not inside one. "Available" is the kernel's own gauge (`kern.memorystatus_level`). The largest job, the original model at full resolution, peaks at 3.2 GB.

`glmocr smoke` is the first thing to run on a new or regenerated page set, on a 24-page preview before the full set is generated: the static check, then 20 pages balanced by level read with `v8d4e4` at 0.7x, with warnings when clean pages do not come back, when the worst level is unreadable or when a page needs more than the token budget. It queues behind a running job, re-reads only pages whose image changed, and never reads a test set.

`--corpus` takes a folder of this repository (`hard`, `held`), a path, or a name under the datasets root (`$GLMOCR_DATASETS`, default `~/Documents/git/datasets`), for example `spark/itdoc_a`. Each set has a role in `datasets.toml`. `glmocr run` refuses to read a `test` set until the run's tag is written in `docs/LEDGER.md` with its question.

Tags in `results/`: `test_*` test-set reads (`docs/LEDGER.md`); `sparkA_*`, `chatgptA_*` dev pilot; `panel_*` the quick panel of the fine-tuning; `fttr*` teacher reads; tags ending in `_op` use the official prompt; `held_*` held-out set; `C_*` tuning set at 0.7x; `S_*` scale sweep; `W_*` lighter variants; `V_*` and `rebuild_*` reproducibility re-runs; `E2_*` the baselines in the environment of `requirements.lock`; `H_*` and `R_*` earlier exploration on `hard/`; `itdocval_*` a set that no longer exists (see `docs/ROADMAP.md`).

## Layout

| Path | What is there |
|---|---|
| `glmocr/` | The package behind the `glmocr` command: `models.py` (checkpoint and variants), `engine.py` (the one place that calls the model), `guard.py` (memory fences), `metrics.py` and `score.py` (scoring), `check.py` (page-set validation), `corpus.py` (paths and roles), `pdf.py` (PDF pages: the text layer or the model) |
| `generators/` | `make_hard.py` for `hard/` and `held/`; `make_sroie.py`, `make_xfund.py`, `make_omnidoc.py`, `make_opb.py`, `make_docatlas.py`, `make_gu.py` convert or build third-party sets into `external/`; `make_f24.py`, `make_photo_set.py`, `augment.py` and `ids.py` build training pages for comb fields (`augment.py` and `make_f24.py` run in their own environment) |
| `experiments/` | Fine-tuning (`ft_data.py`, `ft_targets.py`, `ft_train.py`, `ft_points.py`, `ft_merge.py`, `panel.py`), per-language or per-type comparison (`compare_lang.py`), the retry on saved runs (`loop_retry.py`), the OmniDocBench evaluator (`omnidoc_eval.py`), Omni Parse Bench scoring (`opb_score.py`) and PDF runs (`pdf_eval.py`), dataset audit (`dataset_audit.py`), orientation (`orient_*.py`), reading a page in pieces (`crops.py`) |
| `hard/`, `held/` | Ground truth and metadata of the two synthetic sets |
| `results/` | Raw per-page outputs and timings of every run |
| `tests/` | The scorer on broken inputs, the published tables (`test_tables.py`, and `test_release.py` where the page sets are on the machine), the rules that code can enforce |
| `datasets.toml` | The role of every page set: train, dev, test or spent |
| `NOTICE`, `licenses/` | Attribution and license of third-party code adapted here (Indra OCR, Apache-2.0) |
| `docs/` | `ROADMAP.md` goals and rules, `LEDGER.md` every test-set read, `DATASETS.md` the commissioned sets, `EXTERNAL.md` third-party data and its terms, `DATASET_PROMPT.md` the brief the sets were commissioned with, `MODEL_CARD.md` the card published with the weights. Versions in `CHANGELOG.md` |
| `archive/` | The first, smaller test bed. Nothing in the tables depends on it |
