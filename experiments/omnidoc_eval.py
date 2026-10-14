"""python experiments/omnidoc_eval.py TAG [--bench DIR] : score a saved run on external/omnidoc_200 with the official OmniDocBench evaluator.
The page texts of results/TAG.json become <image name>.md in work/omnidoc_pred/TAG/; the evaluator (end-to-end, quick match) is run with
two metrics only, normalized edit distance on text blocks and on reading order, against external/omnidoc_200/gt.json. Formulas and
tables are left out: their metrics need LaTeX/HTML output and TeX Live, and a page read with "Text Recognition:" writes neither, so a
number for them would measure the prompt, not the model. Output: work/omnidoc_eval/TAG.json with the official averages and the edit
distance of every page (the mean of its text blocks), for paired comparisons.
TAG may be "reference": the annotation's own text blocks as the prediction, to check the pipeline (it must come out near zero).
--bench is the evaluator checkout (github.com/opendatalab/OmniDocBench, commit f133a71) with its own Python 3.11 environment in .venv."""
import argparse, collections, json, os, pathlib, subprocess, statistics
from glmocr import corpus
from glmocr.corpus import ROOT, WORK

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--bench", default="~/Documents/git/bench/OmniDocBench")
a = ap.parse_args()
bench = pathlib.Path(a.bench).expanduser()
src = corpus.corpus_dir("external/omnidoc_200")
meta = json.load(open(src / "meta.json", encoding="utf-8"))
pred = WORK / "omnidoc_pred" / a.tag; pred.mkdir(parents=True, exist_ok=True)
pages = None if a.tag == "reference" else corpus.load_run(a.tag)["pages"]
for m in meta:
    text = (src / f"{m['id']}.txt").read_text(encoding="utf-8") if pages is None else pages[m["id"]]["text"]
    (pred / (pathlib.Path(m["image"]).stem + ".md")).write_text(text, encoding="utf-8")
cfg = {"end2end_eval": {"metrics": {"text_block": {"metric": ["Edit_dist"]}, "reading_order": {"metric": ["Edit_dist"]}},
       "dataset": {"dataset_name": "end2end_dataset", "ground_truth": {"data_path": str(src / "gt.json")},
                   "prediction": {"data_path": str(pred)}, "match_method": "quick_match", "match_workers": 4,
                   "quick_match_truncated_timeout_sec": 300, "match_timeout_sec": 420,
                   "timeout_fallback_max_chunk_span": 10, "timeout_fallback_order_penalty": 0.10}}}
cfgp = WORK / "omnidoc_eval" / f"{a.tag}.yaml"; cfgp.parent.mkdir(parents=True, exist_ok=True)
import yaml; yaml.safe_dump(cfg, open(cfgp, "w"))
subprocess.run([str(bench / ".venv/bin/python"), "pdf_validation.py", "--config", str(cfgp)], cwd=bench, check=True,
               stdout=open(WORK / "omnidoc_eval" / f"{a.tag}.log", "w"), stderr=subprocess.STDOUT)
save = pred.name   # the evaluator names its files after the prediction folder
out = {"tag": a.tag, "evaluator": "OmniDocBench f133a71, end2end, quick_match", "pages": len(meta)}
allres = json.load(open(bench / "result" / f"{save}_quick_match_metric_result.json", encoding="utf-8")) \
    if (bench / "result" / f"{save}_quick_match_metric_result.json").exists() else None
if allres is None:
    cands = sorted((bench / "result").glob(f"{save}*metric_result.json"), key=os.path.getmtime)
    allres = json.load(open(cands[-1], encoding="utf-8"))
for element in ("text_block", "reading_order"):
    out[element] = allres[element]["all"]
    f = sorted((bench / "result").glob(f"{save}*_{element}_result.json"), key=os.path.getmtime)[-1]
    per = collections.defaultdict(list)
    for s in json.load(open(f, encoding="utf-8")):
        per[s["img_id"]].append(s["Edit_dist"] if "Edit_dist" in s else s["edit"])
    out[element + "_per_page"] = {k: statistics.mean(v) for k, v in per.items()}
json.dump(out, open(WORK / "omnidoc_eval" / f"{a.tag}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_per_page")}, ensure_ascii=False)[:1500])
print({k: len(v) for k, v in out.items() if k.endswith("_per_page")})
