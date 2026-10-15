"""work/opb_venv/bin/python experiments/opb_score.py TAG_A TAG_B [...] : saved runs on external/opb_sub scored with Omni Parse Bench's own
scorer (omni-parse-bench 0.2.0), text tests only (present, order, repeat); each run's text goes through the benchmark's markdown.to_html,
as its GLM-OCR adapter does. Prints each run's pass rate, by test origin (derivation) and by suite, then each run after the first against
the first: paired difference of the per-page pass rate, x100, bootstrap over pages (2000, seed 0) and rule 4, as everywhere here.
Tests whose origin is the PDF itself (text_layer, extractor_agreement) favour a reader of the text layer: reported apart."""
import collections, json, pathlib, statistics, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from omni_parse_bench import gold, markdown, score
from glmocr import metrics
from glmocr.score import verdict

TEXT = {"present", "order", "repeat"}; FROM_PDF = {"text_layer", "extractor_agreement"}
D = ROOT / "external/opb_sub"; meta = {m["id"]: m for m in json.load(open(D / "meta.json"))}
tests = {i: [t for t in gold.read(D / f"{i}.gold.json") if t.test_type in TEXT] for i in meta}


def verdicts(tag):
    pages = json.load(open(ROOT / "results" / f"{tag}.json"))["pages"]; out = {}
    for i in meta:
        html = markdown.to_html(pages[i]["text"]) if i in pages else None
        out[i] = {v.id: v.result == "pass" for v in score({"html": html}, tests[i], unsupported={"blocks"})}
    return out


tags = sys.argv[1:]; V = {t: verdicts(t) for t in tags}; origin = {t.id: t.derivation for ts in tests.values() for t in ts}
for t in tags:
    allv = [(i, k, ok) for i, d in V[t].items() for k, ok in d.items()]
    groups = collections.defaultdict(list); suites = collections.defaultdict(list)
    for i, k, ok in allv:
        groups["from the PDF" if origin[k] in FROM_PDF else "independent"].append(ok); suites[meta[i]["suite"]].append(ok)
    print(f"{t:40} all {100 * statistics.mean(ok for *_, ok in allv):5.1f}% of {len(allv)}  |  " +
          "  ".join(f"{g} {100 * statistics.mean(v):.1f}% ({len(v)})" for g, v in sorted(groups.items())))
    print(" " * 42 + "  ".join(f"{s} {100 * statistics.mean(v):.0f}" for s, v in sorted(suites.items())))
page_rate = lambda d, keep=lambda k: True: {i: statistics.mean(ok for k, ok in v.items() if keep(k)) for i, v in d.items() if any(keep(k) for k in v)}
for t in tags[1:]:
    for name, keep in (("all tests", lambda k: True), ("independent tests", lambda k: origin[k] not in FROM_PDF)):
        a, b = page_rate(V[tags[0]], keep), page_rate(V[t], keep); ids = sorted(set(a) & set(b))
        d = lambda s: 100 * statistics.mean(b[i] - a[i] for i in s); lo, hi = metrics.bootstrap(d, ids)
        print(f"{t} - {tags[0]}, {name}: {d(ids):+.1f} [{lo:+.1f}, {hi:+.1f}] {verdict((lo, hi))}  ({len(ids)} pages)")
