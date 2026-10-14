"""python experiments/ft_targets.py TEACHER_TAG [--corpus external/ft_tr] : training answers for a fine-tuning probe -> work/ft/train.jsonl

The ground truth of a commissioned dataset has the right words but not the model's own layout: one table cell per line, where the
model writes one row per line. Trained on that, a model would spend most of its updates learning a new layout. So the answer is
built from both: the WORDS and their order come from the ground truth, the SPACING between two words comes from the model's own
output (the teacher run) wherever the two words are adjacent there too, and from the ground truth elsewhere. What the teacher
skipped (it drops letterheads and footers) or misread is therefore taught; how a page is laid out in text is left alone.
A page where the teacher matched under half of the words is left out: there is no layout to borrow."""
import argparse, difflib, json, os, pathlib, re, unicodedata
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
ap = argparse.ArgumentParser(); ap.add_argument("teacher", nargs="+", help="one or more teacher runs; the first that has a page is used"); ap.add_argument("--corpus", default="external/ft_tr"); ap.add_argument("--min-match", type=float, default=0.5)
ap.add_argument("--errors-only", action="store_true", help="keep a page only if the teacher missed an entity on it or had under --err-match of its words")
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
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); kept = dropped = fine = capped = 0; ratios = []; per_set = {}
with open(out, "w", encoding="utf-8") as f:
    for m in meta:
        if m["id"] not in teacher: continue
        text, ratio = build((corpus / f"{m['id']}.txt").read_text(encoding="utf-8"), teacher[m["id"]]["text"]); ratios.append(ratio)
        if ratio < a.min_match: dropped += 1; continue
        if a.errors_only:   # the original already reads this page: training on it only moves weights that were right
            h = squash(teacher[m["id"]]["text"])
            if ratio >= a.err_match and all(squash(e["text"]) in h for e in m["entities"]): fine += 1; continue
        src = m.get("source", "")
        if a.max_per_set and per_set.get(src, 0) >= a.max_per_set: capped += 1; continue
        per_set[src] = per_set.get(src, 0) + 1; kept += 1
        f.write(json.dumps({"id": m["id"], "images": [str((corpus / f"{m['id']}.png").resolve())], "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Text Recognition:"}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}]}, ensure_ascii=False) + "\n")
ratios.sort(); print(f"{kept} pages -> {out}, {dropped} left out, {fine} left out because the teacher already reads them, {capped} over the cap | per set {per_set} | share of ground-truth words the teacher had: median {ratios[len(ratios)//2]:.2f}, min {ratios[0]:.2f}")
