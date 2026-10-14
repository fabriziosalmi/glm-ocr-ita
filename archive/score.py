"""python score.py tag... : CER vs GT (exact/textlayer pages), CER drift vs bf16 (all pages), speed, memory."""
import json, re, sys, unicodedata, statistics as st
import jiwer
meta = json.load(open("corpus/meta.json"))
def norm(s):  # alnum-only, lowercase: robust to markdown/table markup and whitespace
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-zà-ÿ€$%]", "", s)
def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return jiwer.cer(r, h) if r else 0.0
def load(tag): return json.load(open(f"results/{tag}.json"))
base = load("bf16") if len(sys.argv) < 2 or True else None
print(f"{'tag':14s} {'n':>3s} {'CER_gt':>7s} {'(syn)':>7s} {'(dig)':>7s} {'drift':>7s} {'s/page':>7s} {'prefill':>8s} {'gen t/s':>8s} {'peakGB':>7s}")
for tag in sys.argv[1:]:
    r = load(tag)["pages"]; gt, syn, dig, dr = [], [], [], []
    for m in meta:
        i = m["id"]
        if i not in r: continue
        if m["gt"] != "none":
            ref = open(f"corpus/{i}.txt", encoding="utf-8").read(); c = cer(ref, r[i]["text"]); gt.append(c)
            (syn if i.startswith("syn_") else dig).append(c)
        if i in base["pages"] and tag != "bf16": dr.append(cer(base["pages"][i]["text"], r[i]["text"]))
    f = lambda x: f"{100*st.mean(x):6.2f}%" if x else "     - "
    pg = r.values()
    print(f"{tag:14s} {len(r):3d} {f(gt)} {f(syn)} {f(dig)} {f(dr)} {st.mean(p['s'] for p in pg):7.1f} {st.mean(p['prefill_tps'] for p in pg):8.0f} {st.mean(p['gen_tps'] for p in pg):8.1f} {max(p['peak_gb'] for p in pg):7.2f}")
