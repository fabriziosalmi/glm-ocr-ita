"""python experiments/stage2_confirm.py MODEL BASE_TAG NEW_TAG --corpus NAME [--wait] [--dry]
The stage-2 loop retry (engine.WIDE_PENALTY over WIDE_CTX tokens), applied to a saved run made with --retry-penalty P: only the pages
that stage 2 can change are read again, with the full reader (first read, retry at P, then stage 2); the other pages are copied as they are.
A page can change when its saved retry was still cut by loop_start (second.loop_cut), or when loop_start with the current max_period finds
a repetition in a saved text that the old max_period missed. Decoding is greedy: the first and the second reading of every re-read page
must come out identical to the saved ones, or a prefix of them where a longer max_period cuts a loop sooner, or the environment is not the one that made BASE_TAG and the run stops (--force to go on).
Refuses a test set unless NEW_TAG is in docs/LEDGER.md, as `glmocr run` does."""
import argparse, json, os
from glmocr import corpus
from glmocr.engine import Reader, environment
from glmocr.metrics import loop_start

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("base"); ap.add_argument("tag")
ap.add_argument("--corpus", required=True); ap.add_argument("--wait", action="store_true")
ap.add_argument("--dry", action="store_true", help="only list the pages stage 2 can change"); ap.add_argument("--force", action="store_true")
ap.add_argument("--mem-gb", type=float, default=None)
a = ap.parse_args()
if corpus.role(a.corpus) == "test" and not corpus.ledger_has(a.tag) and not a.dry:
    raise SystemExit(f"'{a.corpus}' is a test set: write the question and the tag `{a.tag}` in docs/LEDGER.md first.")
base = corpus.load_run(a.base)
if not base.get("retry_penalty"):
    raise SystemExit(f"{a.base} was not made with a retry penalty: stage 2 has nothing to follow")


def can_change(p):
    rt = p.get("retry")
    if not rt:
        return False
    second = rt.get("second") or {}
    return (second.get("loop_cut") is not None or bool(second.get("text") and loop_start(second["text"])) or bool(loop_start(p["text"])))


todo = [i for i, p in base["pages"].items() if can_change(p)]
print(f"{len(todo)} pages of {len(base['pages'])} in {a.base} can change under stage 2: {todo}", flush=True)
if a.dry:
    raise SystemExit(0)
out = corpus.result_path(a.tag)
res = json.load(open(out)) if out.exists() else {**{k: v for k, v in base.items() if k != "pages"}, "tag": a.tag, "stage2_of": a.base,
                                                   "env_stage2": environment(), "pages": dict(base["pages"])}
reader = Reader(a.model, max_tokens=base.get("max_tokens", 3000), wait=a.wait, prompt_style=base.get("prompt", "official"),
                retry_penalty=base["retry_penalty"], mem_gb=a.mem_gb)
d = corpus.corpus_dir(a.corpus)
for i in todo:
    if res["pages"][i].get("stage2_read"):
        continue
    old = base["pages"][i]
    new = reader.read(d / f"{i}.png", base.get("scale", 1.0))
    old_second, new_second = (old["retry"].get("second") or {}).get("text"), (new.get("retry", {}).get("second") or {}).get("text")
    common = len(os.path.commonprefix([old_second or "", new_second or ""]))
    same = common == min(len(old_second or ""), len(new_second or ""))   # a retry cut earlier by the longer max_period is a prefix
    if not same and not a.force:
        raise SystemExit(f"{i}: the stage-1 retry does not reproduce the saved one: not the environment of {a.base}")
    page = {k: v for k, v in new.items() if k != "looped"}
    page["stage2_read"] = True
    res["pages"][i] = page
    kept = new.get("retry", {}).get("kept")
    print(f"{i:60.60s} stage1 reproduced={same} (common prefix {common}, saved {len(old_second or "")}, new {len(new_second or "")}) kept={kept} cut={new['loop_cut']} gtok {old['gen_tokens']} -> {new['gen_tokens']} s {old['s']:.0f} -> {new['s']:.0f}", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("->", out)
