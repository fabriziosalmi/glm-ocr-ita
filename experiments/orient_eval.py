"""<python with paddlex and this package> experiments/orient_eval.py [hard|held] : PP-LCNet_x1_0_doc_ori on the set built by orient_make.py -> accuracy by level.
Needs paddlex and the pretrained model in ~/.paddlex/official_models/ (not part of this repo's requirements).
The classifier answers in degrees CLOCKWISE; the labels are counter-clockwise, so correct = (pred + label) % 360 == 0.
To straighten a page: PIL.Image.rotate(int(pred), expand=True)."""
import os, json, sys, time, collections
from glmocr.corpus import ROOT; os.chdir(ROOT)   # every path below is relative to the repository
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddlex import create_model
SRC = sys.argv[1] if len(sys.argv) > 1 else "hard"; D = f"work/orient_{SRC}"
m = create_model("PP-LCNet_x1_0_doc_ori", model_dir=os.path.expanduser("~/.paddlex/official_models/PP-LCNet_x1_0_doc_ori"))
rows = json.load(open(f"{D}/labels.json")); out = []; t = time.time()
for r in rows:
    res = list(m.predict(r["file"], batch_size=1))[0]
    out.append({**r, "pred": int(res["label_names"][0]), "score": float(res["scores"][0])})
s = time.time() - t; json.dump(out, open(f"{D}/pred.json", "w"))
ok = lambda r: (r["pred"] + r["label"]) % 360 == 0; byl = collections.Counter(); n = collections.Counter()
for r in out: byl[r["level"]] += ok(r); n[r["level"]] += 1
print(f"{SRC}: {sum(byl.values())}/{len(out)} correct, {1000*s/len(out):.0f} ms/page | by level: " + " ".join(f"L{l} {byl[l]}/{n[l]}" for l in sorted(n)))
for r in out:
    if not ok(r): print("  wrong:", r["id"], "label", r["label"], "pred", r["pred"], "score", round(r["score"], 2))
