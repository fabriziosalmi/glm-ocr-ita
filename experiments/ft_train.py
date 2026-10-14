"""python experiments/ft_train.py NAME [--data work/ft/train.jsonl] [--iters 200] [--rank 8] [--lr 2e-5] [--mem-gb 11]
Low-rank adapters on the decoder of the original model (bf16); the vision encoder and the projector stay frozen.
-> work/ft/NAME/adapters.safetensors, adapter_config.json, train.json (settings, environment, loss curve)

The whole chain of run op1, the adapter of release v1.0.0, one model job at a time (each step queues behind the one before):
  python experiments/ft_data.py --per-set 8 --out external/ft_tr_op --sets gemini/itdoc_a ... gemini/itdoc_s   # 152 pages
  glmocr run work/glm-ocr-bf16 fttr_op_bf16_1.0 --corpus external/ft_tr_op --wait                             # the teacher: what the model writes today
  python experiments/ft_targets.py fttr_op_bf16_1.0 --corpus external/ft_tr_op --errors-only --max-per-set 6 --out work/ft/train_op1.jsonl
  python experiments/ft_train.py op1 --data work/ft/train_op1.jsonl --iters 90 --lr 1e-5 --save-every 30      # this file, 27 minutes
  python experiments/ft_points.py op1 30 90                                                                   # each saved adapter on the quick panel
  python experiments/ft_merge.py work/ft/op1_90 work/ft/op1_90_merged && glmocr quant v8d8e8 --src work/ft/op1_90_merged --out models/v8d8e8_op1

Only pages whose role in datasets.toml is train may be in --data (ft_data.py refuses anything else). The loss is taken on the
answer only. Batch size is 1: a full-resolution page is 2788 visual tokens, and this machine has 16 GB.
FORMAT. An answer must look in training as it does when the model reads a page: straight after "<|assistant|>\n", and closed
by the token generation stops on. The chat template does neither: for an assistant turn it writes "<|assistant|>\n<think></think>\n"
before the text and nothing after it. Runs step1 and step2 were trained that way (--template-format), so their first steps went
into learning to write "<think></think>", which the model never writes when it reads, and they were never shown where to stop.
CAUSAL MASK. mlx-vlm 0.7.6 calls the model in training with the batch's attention mask (all ones for one page). GLM-OCR's decoder
builds its causal mask only when it is given none, so with that call every token sees the tokens after it. Measured on the original
model, answer = its own output: mean loss 1.6 to 3.1 and first choice right on 54 to 71% of the tokens with the trainer's call,
0.001 to 0.04 and 99 to 100% with no mask. An adapter trained that way learns to copy the next token from the future, which is why
runs step1 to step3 fell to a loss near zero in a few dozen steps and read no better. The loss below passes no mask
(--trainer-mask reproduces the old call). One page per step, so there is no padding to mask.
Memory: same fences as every model job (glmocr/guard.py), with a higher ceiling because gradients are kept; the watch runs
before every training page and stops the job, saving the adapter as it is.
"""
import argparse, json, os, pathlib, random, time
from glmocr import guard
from glmocr.corpus import ROOT, SOURCE, WORK; os.chdir(ROOT)   # every path below is relative to the repository
from glmocr.engine import environment, official_prompt

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--data", default="work/ft/train.jsonl")
ap.add_argument("--iters", type=int, default=200); ap.add_argument("--rank", type=int, default=8); ap.add_argument("--alpha", type=float, default=16)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--max-seq", type=int, default=4608, help="visual tokens + prompt + answer; longer pages are left out")
ap.add_argument("--mem-gb", type=float, default=10.0, help="ceiling of the job; the rehearsal peaked at 7.1 GB")
ap.add_argument("--need-gb", type=float, default=7.5, help="memory the system must have available before the model is loaded, floor apart")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--swap-gb", type=float, default=1.5, help="stop if swap in use grows by more than this during the run")
ap.add_argument("--eos", default="<|user|>", help="what closes an answer: the token the model itself stops on when it reads a page")
ap.add_argument("--template-format", action="store_true", help="the format of runs step1 and step2, kept to reproduce them: see FORMAT below")
ap.add_argument("--prompt", choices=["official", "mlx"], default="official", help="official: the prompt of the reference implementation, as the reader uses; mlx: with /nothink and <think></think>, as runs step1 to step4")
ap.add_argument("--trainer-mask", action="store_true", help="the loss call of runs step1 to step3, kept to reproduce them: see CAUSAL MASK below")
ap.add_argument("--dry-run", action="store_true", help="check the data and the settings, load nothing")
a = ap.parse_args()
out = WORK / "ft" / a.name
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
missing = [r["id"] for r in rows if not all(pathlib.Path(i).exists() for i in r["images"])]
if missing: raise SystemExit(f"{len(missing)} pages of {a.data} have no image, e.g. {missing[0]}")
random.Random(a.seed).shuffle(rows)
print(f"{len(rows)} pages in {a.data}; {a.iters} steps of one page, rank {a.rank}, lr {a.lr}, ceiling {a.mem_gb} GB -> {out}")
if a.dry_run: raise SystemExit(0)
if (out / "adapters.safetensors").exists(): raise SystemExit(f"{out} already holds an adapter: choose another name")

held = guard.lock(WORK / ".model.lock", wait=True)
guard.preflight(SOURCE, a.mem_gb)
have = guard.available_gb()
if have < a.need_gb + guard.FLOOR_GB:   # training keeps gradients: it needs far more than the weights, and a refusal is better than swapping
    raise SystemExit(f"training needs about {a.need_gb:.1f} GB plus a floor of {guard.FLOOR_GB:.1f} GB; the system has {have:.1f} GB available. Close something first")
print(f"memory: {have:.1f} GB available, ceiling {a.mem_gb:.1f} GB")
guard.apply_limits(a.mem_gb); swap0 = guard.swap_used_gb()
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_vlm.utils import load
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model, print_trainable_parameters, save_adapter

mx.random.seed(a.seed)
model, proc = load(str(SOURCE), processor_config={"trust_remote_code": True})
model = get_peft_model(model, find_all_linear_names(model.language_model), rank=a.rank, alpha=a.alpha, dropout=0.0, verbose=False)
print_trainable_parameters(model)


if not a.template_format:
    import mlx_vlm.trainer.datasets as D
    _template = D.apply_chat_template
    def as_when_reading(processor, config, conversation, add_generation_prompt=False, **kw):
        """prompt as at inference + answer + end token, instead of the template's rendering of an assistant turn"""
        fix = official_prompt if a.prompt == "official" else (lambda p: p)
        last = conversation[-1] if isinstance(conversation, list) and conversation else None
        if add_generation_prompt or not isinstance(last, dict) or last.get("role") != "assistant":
            out = _template(processor, config, conversation, add_generation_prompt=add_generation_prompt, **kw)
            return fix(out) if add_generation_prompt else out
        c = last["content"]
        answer = c if isinstance(c, str) else "".join(x.get("text", "") for x in c if x.get("type") == "text")
        return fix(_template(processor, config, conversation[:-1], add_generation_prompt=True, **kw)) + answer.strip() + a.eos
    D.apply_chat_template = as_when_reading


def causal_loss(model, batch, train_on_completions=False, assistant_id=None):
    """Cross-entropy on the answer, the model called as when it reads: no mask passed, so the decoder builds its causal one."""
    ids = batch["input_ids"]
    assert ids.shape[0] == 1, "one page per step: a batch would need a causal mask combined with padding"
    kw = {k: v for k, v in batch.items() if k not in ("input_ids", "pixel_values", "attention_mask", "completion_mask")}
    model.language_model._position_ids = None   # positions are cached on the model between calls
    logits = model(ids[:, :-1], batch["pixel_values"], None, **kw).logits.astype(mx.float32)
    labels = ids[:, 1:]
    if logits.shape[1] > labels.shape[1]: logits = logits[:, -labels.shape[1]:, :]
    weights = batch["completion_mask"][:, 1:].astype(mx.float32)
    return (nn.losses.cross_entropy(logits, labels) * weights).sum() / mx.maximum(weights.sum(), 1)


class Watched(VisionDataset):
    """The trainer's dataset, with the memory watch before every page and pages over --max-seq left out."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); self.skipped = []; self.seen = 0
    def process(self, item):
        try:
            guard.watch(mx.get_peak_memory() / guard.GB, a.mem_gb)
            grown = guard.swap_used_gb() - swap0
            if grown > a.swap_gb: raise guard.MemoryStop(f"swap in use grew by {grown:.1f} GB since the start: the machine is short of memory")
        except guard.MemoryStop:
            save_adapter(model, out / "adapters.safetensors"); raise
        mx.reset_peak_memory(); self.seen += 1
        return super().process(item)


data = Watched(rows, model.config.__dict__, proc, train_on_completions=True)
lengths = {}
keep = []
for r in rows:   # token length of every page, before the first step: nothing is truncated silently
    n = int(VisionDataset.process(data, r)["input_ids"].size)
    lengths[r["id"]] = n
    (keep if n <= a.max_seq else data.skipped).append(r)
if data.skipped: print(f"{len(data.skipped)} pages are longer than {a.max_seq} tokens and are left out (longest {max(lengths.values())})")
data.dataset = keep
out.mkdir(parents=True, exist_ok=True)
log = {"name": a.name, "data": a.data, "pages": len(keep), "left_out": [r["id"] for r in data.skipped], "iters": a.iters, "rank": a.rank,
       "alpha": a.alpha, "lr": a.lr, "format": "template" if a.template_format else "as when reading", "prompt": a.prompt, "loss": "trainer mask (future visible)" if a.trainer_mask else "causal", "eos": None if a.template_format else a.eos, "save_every": a.save_every, "max_seq": a.max_seq, "seed": a.seed, "env": environment(), "tokens_longest": max(lengths.values())}
json.dump(log, open(out / "train.json", "w"), indent=1)
t = time.time()
from mlx_vlm.trainer.sft_trainer import vision_language_loss_fn
train(model=model, optimizer=optim.Adam(learning_rate=a.lr), train_dataset=data, val_dataset=None, train_on_completions=True,
      loss_fn=vision_language_loss_fn if a.trainer_mask else causal_loss,
      args=TrainingArgs(batch_size=1, iters=a.iters, steps_per_report=10, steps_per_eval=10 ** 9, steps_per_save=a.save_every, val_batches=0,
                        max_seq_length=a.max_seq, adapter_file=str(out / "adapters.safetensors"), grad_checkpoint=True, learning_rate=a.lr))
log.update(minutes=(time.time() - t) / 60, peak_gb=mx.get_peak_memory() / guard.GB)
json.dump(log, open(out / "train.json", "w"), indent=1)
print(f"done in {log['minutes']:.0f} min -> {out}")
