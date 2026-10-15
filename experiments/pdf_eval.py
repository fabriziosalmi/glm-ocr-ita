"""python experiments/pdf_eval.py TAG --corpus NAME [--mode auto] [--model models/v8d8e8_op1] [--retry-penalty 1.1] : a page set read from
its PDFs (meta "pdf") through glmocr/pdf.py -> results/TAG.json, in the format of `glmocr run`, so that `glmocr compare` and `score`
work on it. Pages read from the text layer have no tokens and cannot loop. The model is loaded only if a page needs it.
Refuses a test set unless TAG is in docs/LEDGER.md, as `glmocr run` does."""
import argparse, json
from glmocr import corpus
from glmocr.pdf import read_pdf

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--corpus", required=True); ap.add_argument("--mode", default="auto")
ap.add_argument("--model", default="models/v8d8e8_op1"); ap.add_argument("--retry-penalty", type=float, default=1.1)
ap.add_argument("--fallback", help="a saved run of the model on the same renders: its reading is taken for a page whose layer fails a check, instead of reading it again")
a = ap.parse_args()
if corpus.role(a.corpus) == "test" and not corpus.ledger_has(a.tag):
    raise SystemExit(f"'{a.corpus}' is a test set: write the question and the tag `{a.tag}` in docs/LEDGER.md first.")
d = corpus.corpus_dir(a.corpus); meta = corpus.load_meta(a.corpus); reader = []
fallback = corpus.load_run(a.fallback)["pages"] if a.fallback else None

class Saved:   # stands in for the model: the page's reading from the fallback run
    def __init__(self): self.id = None
    def read(self, image): return dict(fallback[self.id])
saved = Saved()
def make():
    from glmocr.engine import Reader
    reader.append(Reader(a.model, retry_penalty=a.retry_penalty, wait=True)); return reader[0]
pages = {}
for m in meta:
    saved.id = m["id"]
    (r,) = read_pdf(d / m["pdf"], mode=a.mode, reader=saved if fallback else None, make_reader=lambda: reader[0] if reader else make())
    pages[m["id"]] = {"gen_tokens": 0, "loop_cut": None, "looped": False, **r}
    print(a.tag, m["id"], r["source"], r["reason"], f"{r['s']:.2f}s")
json.dump({"tag": a.tag, "model": a.model if reader else f"text layer, else {a.fallback}" if fallback else "text layer", "corpus": a.corpus, "mode": a.mode, "pages": pages},
          open(corpus.result_path(a.tag), "w"), ensure_ascii=False, indent=1)
print("->", corpus.result_path(a.tag))
