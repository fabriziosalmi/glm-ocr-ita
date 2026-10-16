"""python experiments/ft_targets.py TEACHER_TAG [--corpus external/ft_tr] : training answers for a fine-tuning probe -> work/ft/train.jsonl

The ground truth of a commissioned dataset has the right words but not the model's own layout: one table cell per line, where the
model writes one row per line. Trained on that, a model would spend most of its updates learning a new layout. So the answer is
built from both: the WORDS and their order come from the ground truth, the SPACING between two words comes from the model's own
output (the teacher run) wherever the two words are adjacent there too, and from the ground truth elsewhere. What the teacher
skipped (it drops letterheads and footers) or misread is therefore taught; how a page is laid out in text is left alone.
A page where the teacher matched under half of the words is left out: there is no layout to borrow.
A page whose ground truth does not hold what the page shows is left out too: training on it would teach the model to skip that
text. Four checks against the teacher's reading:
- the share of its text absent from the ground truth (letters and digits only, runs of five, each counted as often as it occurs,
  so spacing does not count) above --max-missing. The share includes the teacher's own misreads, so the bar cannot be low:
  degraded F24 pages sit at 0.11 to 0.16 with every printed label in their ground truth; the worst DocAtlas pages, 0.25 to 0.41;
- tick boxes, or blank lines (dots, underscores), read at least twice more than the ground truth holds. They are few characters:
  a page missing all its box options would still sit near 0.05 on the first check;
- the share of its letters and digits that the ground truth holds in the same order, under --min-order: an answer in another
  order than the page's teaches the model to move text. Letters and digits only, so spacing does not count. Calibrated on the
  data of op1 to op4 (2026-10-15): median 1.00 on the gemini, Gazzetta and realistic pages, 0.89 to 0.91 on the F24 pages,
  and down to 0.62 on DocAtlas pages laid out as one table cell. 0.80 leaves out 4 of 152 gemini pages, 2 of 24 F24 and 5 of
  60 DocAtlas;
- a line of the answer over 1000 characters and more than twice the teacher's longest line: a form flattened into one table cell
  (one DocAtlas page of op3's data, 1313 against 403); long prose paragraphs, which the teacher writes on one line too, pass.
  Off with --min-order 0, as the order check.
`--max-missing 2 --min-order 0` gives back the answers of op1 and of op2's comb-field pages, made before these checks.
--tail-only (2026-10-16, for op5): keep only the pages where the teacher stops before the ground truth's last line
(metrics.last_line_missing), the habit found on ocr-bench-v1; the answer holds that line, as every answer holds the ground truth's
words. The summary counts those pages among the kept ones in any case (op1: 28 of 91). For sets whose ground truth is in reading
order only.
"""
import argparse, collections, difflib, json, os, pathlib, re, unicodedata
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
from glmocr.metrics import form_gap, in_order, last_line_missing, text_missing as missing
ap = argparse.ArgumentParser(); ap.add_argument("teacher", nargs="+", help="one or more teacher runs; the first that has a page is used"); ap.add_argument("--corpus", default="external/ft_tr"); ap.add_argument("--min-match", type=float, default=0.5)
ap.add_argument("--errors-only", action="store_true", help="keep a page only if the teacher missed an entity on it or had under --err-match of its words")
ap.add_argument("--min-order", type=float, default=0.80, help="smallest share of the teacher's letters and digits found in the same order in the ground truth")
ap.add_argument("--max-missing", type=float, default=0.20, help="largest share of the teacher's text (5-character runs of letters and digits) absent from the ground truth")
ap.add_argument("--tail-only", action="store_true", help="keep only pages where the teacher misses the ground truth's last line")
ap.add_argument("--err-match", type=float, default=0.90); ap.add_argument("--max-per-set", type=int, default=None); ap.add_argument("--out", default="work/ft/train.jsonl")
a = ap.parse_args(); corpus = pathlib.Path(a.corpus)
teacher = {}
for t in reversed(a.teacher): teacher.update(json.load(open(f"results/{t}.json"))["pages"])
meta = json.load(open(corpus / "meta.json"))
squash = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).lower())
def split(text):  # words and the whitespace that precedes each one
    parts = re.findall(r"(\s*)(\S+)", text.strip()); return [w for _, w in parts], [s for s, _ in parts]
key = lambda w: unicodedata.normalize("NFKC", w).lower()
def build(gt, hyp):
    gw, gs = split(gt); hw, hs = split(hyp); to_h = {}; same = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [key(w) for w in gw], [key(w) for w in hw], autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):   # a misread word still sits in the same place
            for k in range(i2 - i1): to_h[i1 + k] = j1 + k
            same += (i2 - i1) if tag == "equal" else 0
    out = []
    for j, w in enumerate(gw):
        if j: out.append(hs[to_h[j]] if j in to_h and j - 1 in to_h and to_h[j] == to_h[j - 1] + 1 else gs[j])
        out.append(w)
    return "".join(out), same / max(len(gw), 1)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); kept = dropped = fine = capped = incomplete = disordered = tails = untailed = 0; ratios = []; per_set = {}
with open(out, "w", encoding="utf-8") as f:
    for m in meta:
        if m["id"] not in teacher: continue
        gt = (corpus / f"{m['id']}.txt").read_text(encoding="utf-8")
        text, ratio = build(gt, teacher[m["id"]]["text"]); ratios.append(ratio)
        if ratio < a.min_match: dropped += 1; continue
        if missing(gt, teacher[m["id"]]["text"]) > a.max_missing or max(form_gap(gt, teacher[m["id"]]["text"]).values()) >= 2: incomplete += 1; continue
        if a.min_order and in_order(gt, teacher[m["id"]]["text"]) < a.min_order: disordered += 1; continue
        longest = max(map(len, text.split("\n")))
        if a.min_order and longest > 1000 and longest > 2 * max(map(len, teacher[m["id"]]["text"].split("\n"))): disordered += 1; continue
        if a.errors_only:   # the original already reads this page: training on it only moves weights that were right
            h = squash(teacher[m["id"]]["text"])
            if ratio >= a.err_match and all(squash(e["text"]) in h for e in m["entities"]): fine += 1; continue
        tail = last_line_missing(gt, teacher[m["id"]]["text"])
        if a.tail_only and not tail: untailed += 1; continue
        src = m.get("source", "")
        if a.max_per_set and per_set.get(src, 0) >= a.max_per_set: capped += 1; continue
        per_set[src] = per_set.get(src, 0) + 1; kept += 1; tails += tail
        f.write(json.dumps({"id": m["id"], "images": [str((corpus / f"{m['id']}.png").resolve())], "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Text Recognition:"}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}]}, ensure_ascii=False) + "\n")
ratios.sort(); print(f"{kept} pages -> {out}, {dropped} left out, {incomplete} left out because their ground truth lacks what the teacher read, {disordered} because it is in another order or other lines than the teacher's reading, {fine} left out because the teacher already reads them, {capped} over the cap{f", {untailed} because the teacher reads their last line" if a.tail_only else ""} | {tails} of the kept pages teach a last line the teacher missed | per set {per_set} | share of ground-truth words the teacher had: median {ratios[len(ratios)//2]:.2f}, min {ratios[0]:.2f}")
