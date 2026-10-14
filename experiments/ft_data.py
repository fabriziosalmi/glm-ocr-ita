"""python experiments/ft_data.py [--per-set 40] [--hold 10] [--levels 1 2 3] : the training pages of a fine-tuning run.
-> external/ft_tr/ : --per-set pages from each of --sets, which must have the role train in datasets.toml, in the format glmocr reads
   (symlinks to the pages). The answers are built afterwards by ft_targets.py.
Pages come from each set's sampling order (glmocr/corpus.py), so the choice depends on the page ids alone. The first --hold
pages of that order are never taken: they are the set's share of the panel (experiments/panel.py), read and never trained on.
Levels above 3 are left out on purpose: on the worst scans part of the text cannot be read by a person either, and training a
model to write text it cannot see teaches it to invent."""
import argparse, json, os, pathlib
from glmocr.corpus import ROOT, corpus_dir, load_meta, registry, sampling_order, Missing; os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--per-set", type=int, default=40); ap.add_argument("--hold", type=int, default=10)
ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--out", default="external/ft_tr")
ap.add_argument("--sets", nargs="+", default=[f"gemini/itdoc_{c}" for c in "abcdef"], help="train sets to draw from (default: the six of the first run)")
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); pages = []
for old in out.iterdir():
    if old.is_symlink() or old.name == "meta.json": old.unlink()
for name in a.sets:
    if registry().get(name, {}).get("role") != "train": raise SystemExit(f"{name} is not a train set in datasets.toml: weights may not see it")
    try: src = corpus_dir(name)
    except Missing: print(name, "not on this machine, skipped"); continue
    order = sampling_order(load_meta(name)); held = {m["id"] for m in order[:a.hold]}
    take = [m for m in order[a.hold:] if m["level"] in a.levels][:a.per_set]
    assert not held & {m["id"] for m in take}
    for m in take:
        for e in m["entities"]: e["type"] = e.get("type") or "untyped"
        for ext in ("png", "txt"): os.symlink(src / f"{m['id']}.{ext}", out / f"{m['id']}.{ext}")
        pages.append({**m, "source": name})
    print(f"{name}: {len(take)} pages, held back {len(held)}", {l: sum(m["lang"] == l for m in take) for l in ("it", "en")}, {l: sum(m["level"] == l for m in take) for l in a.levels})
assert len({m["id"] for m in pages}) == len(pages), "page ids clash between sets"
json.dump(pages, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
print(len(pages), "training pages ->", out)
