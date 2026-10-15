"""python experiments/omnidoc_eval.py TAG [--corpus external/omnidoc_200] [--full] [--bench DIR] : score a saved run on an OmniDocBench
subset (generators/make_omnidoc.py) with the official evaluator. The page texts of results/TAG.json become <image name>.md in
work/omnidoc_pred/TAG/; the evaluator (end-to-end, quick match) runs against the subset's gt.json.
Metrics: normalized edit distance on text blocks and on reading order; with --full (from 2026-10-15) also tables (TEDS and edit
distance) and display formulas (edit distance only: the official CDM needs TeX Live, left for later by the owner's choice, and a
formula number here is said to be an edit distance). With --full the output also has "overall_edit", the benchmark's overall
score ((1 - text edit) x 100 + table TEDS + formula score) / 3 with (1 - formula edit) x 100 in place of CDM: not the official
overall, and named so. A page read whole with "Text Recognition:" writes tables as markdown or HTML and formulas as plain text
or LaTeX as it pleases: what --full measures is the reader as used, prompt included.
Output: work/omnidoc_eval/TAG.json with the official averages and each page's mean per element, for paired comparisons.
TAG may be "reference": the annotation itself as the prediction (text blocks and titles, and with --full its tables in HTML and
formulas in LaTeX, in annotation order), to check the pipeline: it must come out near zero.
--bench is the evaluator checkout (github.com/opendatalab/OmniDocBench, commit f133a71) with its own Python 3.11 environment in .venv."""
import argparse, collections, json, os, pathlib, subprocess, statistics
from glmocr import corpus
from glmocr.corpus import ROOT, WORK

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--bench", default="~/Documents/git/bench/OmniDocBench")
ap.add_argument("--corpus", default="external/omnidoc_200"); ap.add_argument("--full", action="store_true", help="also tables and display formulas")
a = ap.parse_args()
bench = pathlib.Path(a.bench).expanduser()
src = corpus.corpus_dir(a.corpus)
meta = json.load(open(src / "meta.json", encoding="utf-8"))
name = a.tag if a.tag != "reference" else f"reference_{src.name}" + ("_full" if a.full else "")
pred = WORK / "omnidoc_pred" / name; pred.mkdir(parents=True, exist_ok=True)
pages = None if a.tag == "reference" else corpus.load_run(a.tag)["pages"]
gt = {pathlib.Path(p["page_info"]["image_path"]).stem: p for p in json.load(open(src / "gt.json", encoding="utf-8"))}


def reference(stem):
    """the annotation as markdown, in its own order"""
    out = []
    for d in sorted(gt[stem]["layout_dets"], key=lambda d: int(d["order"]) if str(d.get("order", "")).lstrip("-").isdigit() else 10 ** 6):
        if d["category_type"] in ("text_block", "title") and d.get("text"):
            out.append(d["text"])
        elif a.full and d["category_type"] == "table" and d.get("html"):
            out.append(d["html"])
        elif a.full and d["category_type"] == "equation_isolated" and d.get("latex"):
            out.append(d["latex"].strip() if d["latex"].strip().startswith("$$") else f"$$\n{d['latex'].strip()}\n$$")
    return "\n\n".join(out)


for m in meta:
    stem = pathlib.Path(m["image"]).stem
    text = (reference(stem) if a.full else (src / f"{m['id']}.txt").read_text(encoding="utf-8")) if pages is None else pages[m["id"]]["text"]
    (pred / (stem + ".md")).write_text(text, encoding="utf-8")
metrics = {"text_block": {"metric": ["Edit_dist"]}, "reading_order": {"metric": ["Edit_dist"]}}
if a.full:
    metrics |= {"display_formula": {"metric": ["Edit_dist"]}, "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 4}}
cfg = {"end2end_eval": {"metrics": metrics,
       "dataset": {"dataset_name": "end2end_dataset", "ground_truth": {"data_path": str(src / "gt.json")},
                   "prediction": {"data_path": str(pred)}, "match_method": "quick_match", "match_workers": 4,
                   "quick_match_truncated_timeout_sec": 300, "match_timeout_sec": 420,
                   "timeout_fallback_max_chunk_span": 10, "timeout_fallback_order_penalty": 0.10}}}
cfgp = WORK / "omnidoc_eval" / f"{name}.yaml"; cfgp.parent.mkdir(parents=True, exist_ok=True)
import yaml; yaml.safe_dump(cfg, open(cfgp, "w"))
subprocess.run([str(bench / ".venv/bin/python"), "pdf_validation.py", "--config", str(cfgp)], cwd=bench, check=True,
               stdout=open(WORK / "omnidoc_eval" / f"{name}.log", "w"), stderr=subprocess.STDOUT)
save = pred.name   # the evaluator names its files after the prediction folder
out = {"tag": a.tag, "corpus": a.corpus, "evaluator": "OmniDocBench f133a71, end2end, quick_match", "pages": len(meta)}
allres = json.load(open(bench / "result" / f"{save}_quick_match_metric_result.json", encoding="utf-8")) \
    if (bench / "result" / f"{save}_quick_match_metric_result.json").exists() else None
if allres is None:
    cands = sorted((bench / "result").glob(f"{save}*metric_result.json"), key=os.path.getmtime)
    allres = json.load(open(cands[-1], encoding="utf-8"))
for element in metrics:
    out[element] = allres[element]["all"]
    f = sorted((bench / "result").glob(f"{save}*_{element}_result.json"), key=os.path.getmtime)[-1]
    per = collections.defaultdict(list)
    for s in json.load(open(f, encoding="utf-8")):
        per[s["img_id"]].append(s["Edit_dist"] if "Edit_dist" in s else s["edit"])
    out[element + "_per_page"] = {k: statistics.mean(v) for k, v in per.items()}
if a.full:
    text_ed = out["text_block"]["Edit_dist"]["ALL_page_avg"]; teds = out["table"]["TEDS"]["all"]; form_ed = out["display_formula"]["Edit_dist"]["ALL_page_avg"]
    out["overall_edit"] = ((1 - text_ed) * 100 + teds * 100 + (1 - form_ed) * 100) / 3
json.dump(out, open(WORK / "omnidoc_eval" / f"{name}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_per_page")}, ensure_ascii=False)[:1500])
print({k: len(v) for k, v in out.items() if k.endswith("_per_page")})
