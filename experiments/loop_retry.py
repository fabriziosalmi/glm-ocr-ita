"""python experiments/loop_retry.py MODEL BASE_TAG NEW_TAG --corpus NAME --penalty P [--wait]
The retry policy of the reader (engine.Reader with retry_penalty=P), applied to a saved run: every page of BASE_TAG that looped is
read again with repetition penalty P, and engine.keep_retry decides which reading stays; the other pages are copied as they are.
Decoding is greedy and deterministic, so this gives what `glmocr run --retry-penalty P` would, in the same environment, for the
price of the looping pages only. Refuses a test set unless NEW_TAG is in docs/LEDGER.md, as `glmocr run` does."""
import argparse, json
from glmocr import corpus
from glmocr.engine import Reader, environment, keep_retry
from glmocr.metrics import is_loop

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("base"); ap.add_argument("tag")
ap.add_argument("--corpus", required=True); ap.add_argument("--penalty", type=float, required=True); ap.add_argument("--wait", action="store_true")
a = ap.parse_args()
if corpus.role(a.corpus) == "test" and not corpus.ledger_has(a.tag):
    raise SystemExit(f"'{a.corpus}' is a test set: write the question and the tag `{a.tag}` in docs/LEDGER.md first.")
base = corpus.load_run(a.base)
looped = lambda p: p.get("loop_cut") is not None or is_loop(p["text"], p["gen_tokens"])
todo = [i for i, p in base["pages"].items() if looped(p)]
out = corpus.result_path(a.tag)
res = json.load(open(out)) if out.exists() else {**{k: v for k, v in base.items() if k != "pages"}, "tag": a.tag, "retry_penalty": a.penalty,
                                                   "retry_of": a.base, "env_retry": environment(), "pages": dict(base["pages"])}
print(f"{len(todo)} looping pages of {len(base['pages'])} in {a.base}, read again with penalty {a.penalty}")
reader = Reader(a.model, max_tokens=base.get("max_tokens", 3000), wait=a.wait, prompt_style=base.get("prompt", "official"))
d = corpus.corpus_dir(a.corpus)
for i in todo:
    if "retry" in res["pages"][i]:
        continue
    first = base["pages"][i]
    second = reader.read(d / f"{i}.png", base.get("scale", 1.0), penalty=a.penalty)
    first_l = dict(first, looped=True)
    kept = keep_retry(first_l, second)
    page = {k: v for k, v in (second if kept else first).items() if k not in ("looped", "retry")}
    page["retry"] = {"penalty": a.penalty, "kept": kept,
                     "second": {k: second[k] for k in ("text", "loop_cut", "s", "gen_tokens", "looped")}}
    page["s"] = first["s"] + second["s"]
    res["pages"][i] = page
    print(f"{i:28s} second {'loops' if second['looped'] else 'ends'} after {second['gen_tokens']} tokens; kept {'second' if kept else 'first'}", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("->", out)
