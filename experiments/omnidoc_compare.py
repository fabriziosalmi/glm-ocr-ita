"""python experiments/omnidoc_compare.py TAG_A TAG_B [--corpus external/omnidoc_200] [--by-type] : the paired difference A - B of
OmniDocBench edit distance, times 100 (positive = B better), per language of the subset (en, zh, and mix where there is one), from
work/omnidoc_eval/TAG.json (experiments/omnidoc_eval.py). Every per-page element both runs have is compared: text blocks and reading
order, and with --full evaluations also display formulas and tables (both by edit distance; the table TEDS is a whole-set number in the
evaluator's output, printed with overall_edit, not bootstrapped). Pages are those with matched blocks on both sides; bootstrap over pages
(2000, seed 0) as everywhere else; verdict by rule 4 (score.verdict, margin -1.5). --by-type adds the text blocks per page type
(the subset's "tpl"), descriptive only: some types have a handful of pages."""
import argparse, json, statistics
from glmocr import corpus, metrics, score
from glmocr.corpus import WORK

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--corpus", default="external/omnidoc_200"); ap.add_argument("--by-type", action="store_true")
args = ap.parse_args()
meta = {m["id"]: m for m in corpus.load_meta(args.corpus)}   # the evaluator keys pages by image file name
page = lambda i: meta.get(i.rsplit(".", 1)[0])
ea, eb = (json.load(open(WORK / "omnidoc_eval" / f"{t}.json", encoding="utf-8")) for t in (args.a, args.b))
print(f"A - B (x100, positive = B better), A = {args.a}, B = {args.b}, {args.corpus}")
for t, e in ((args.a, ea), (args.b, eb)):
    if "overall_edit" in e:
        print(f"  {t}: overall_edit {e['overall_edit']:.1f}, table TEDS {100 * e['table']['TEDS']['all']:.1f}, "
              f"text {100 * e['text_block']['Edit_dist']['ALL_page_avg']:.1f}, formula {100 * e['display_formula']['Edit_dist']['ALL_page_avg']:.1f} (edit x100)")


def compare(metric, label, keep):
    ids = [i for i in ea[metric] if i in eb[metric] and page(i) and keep(page(i))]
    if not ids:
        return
    d = lambda s: 100 * statistics.mean(ea[metric][i] - eb[metric][i] for i in s)
    mean = lambda e: 100 * statistics.mean(e[metric][i] for i in ids)
    lo, hi = metrics.bootstrap(d, ids)
    print(f"{metric[:-9]:15} {label:14} {len(ids):3} pages  A {mean(ea):5.1f}  B {mean(eb):5.1f}  diff {d(ids):+5.1f} [{lo:+.1f}, {hi:+.1f}]  {score.verdict((lo, hi))}")


langs = sorted({m["lang"] for m in meta.values()}, key=["en", "zh", "mix"].index)
for metric in [k for k in ea if k.endswith("_per_page") and k in eb]:
    for lang in langs:
        compare(metric, lang, lambda m, lang=lang: m["lang"] == lang)
    compare(metric, "all", lambda m: True)
if args.by_type:
    for tpl in sorted({m["tpl"] for m in meta.values()}):
        compare("text_block_per_page", tpl, lambda m, tpl=tpl: m["tpl"] == tpl)
