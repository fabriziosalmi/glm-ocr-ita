"""glmocr <command>: every step from the upstream checkpoint to the tables in docs/RESULTS.md."""
import argparse
import json
import sys

from . import corpus, guard, metrics
from .models import RELEASE


def cmd_prepare(a):
    from . import models
    print("ok", models.prepare(), "revision", models.REVISION[:7])


def cmd_quant(a):
    from . import models
    dst, mb = models.quantize(a.variant, a.out, a.src)
    print(f"done {a.variant} -> {dst}  {mb:.0f} MB")


def cmd_infer(a):
    from .engine import Reader
    make = lambda: Reader(a.model, max_tokens=a.max_tokens, mem_gb=a.mem_gb, prompt_style=a.prompt, retry_penalty=a.retry_penalty)
    if a.image.lower().endswith(".pdf"):
        return infer_pdf(a, make)
    r = make().read(a.image, a.scale)
    print(r["text"])
    print(f"{r['s']:.1f} s, {r['prompt_tokens']} prompt tokens, {r['gen_tokens']} generated"
          + (" - REPETITION LOOP, page not read" if r["looped"] else ""), file=sys.stderr)
    if a.output:
        json.dump(r, open(a.output, "w"), ensure_ascii=False, indent=1)
    return 2 if r["looped"] else 0


def infer_pdf(a, make_reader):
    """Every page of a PDF: the text layer where it passes the checks of glmocr/pdf.py (mode auto), the model elsewhere."""
    from .pdf import read_pdf
    pages, looped = [], False
    for r in read_pdf(a.image, mode=a.mode, pages=a.pages, make_reader=make_reader):
        print(("\f" if pages else "") + r["text"])
        how = "text layer" if r["source"] == "text" else f"model ({r['reason']})" if r["reason"] != "mode ocr" else "model"
        print(f"page {r['page']}: {how}, {r['s']:.{2 if r['source'] == 'text' else 1}f} s" + (" - REPETITION LOOP, page not read" if r.get("looped") else ""), file=sys.stderr)
        pages.append(r); looped |= bool(r.get("looped"))
    if a.output:
        json.dump(pages, open(a.output, "w"), ensure_ascii=False, indent=1)
    return 2 if looped else 0


def cmd_run(a):
    """A test set is read once per decision: the tag must be written in docs/LEDGER.md, with its question, before the run."""
    role = corpus.role(a.corpus)
    if role == "test" and not corpus.ledger_has(a.tag):
        raise SystemExit(f"'{a.corpus}' is a test set (datasets.toml). Write the question and the tag `{a.tag}` in docs/LEDGER.md first.")
    from .engine import Reader, environment
    meta = corpus.load_meta(a.corpus)
    d = corpus.corpus_dir(a.corpus)
    out = corpus.result_path(a.tag)
    out.parent.mkdir(exist_ok=True)
    reader = Reader(a.model, adapter=a.adapter, max_tokens=a.max_tokens, mem_gb=a.mem_gb, wait=a.wait, prompt_style=a.prompt, retry_penalty=a.retry_penalty)
    setup = {"model": a.model, "corpus": corpus.split_name(a.corpus)[0], "scale": a.scale, "max_tokens": a.max_tokens, "adapter": a.adapter, "prompt": a.prompt, **({"retry_penalty": a.retry_penalty} if a.retry_penalty else {}), "env": environment()}
    res = json.load(open(out)) if out.exists() else {"tag": a.tag, **setup, "load_s": reader.load_s, "pages": {}}
    res.setdefault("prompt", "mlx")   # files written before 2026-10-14 used mlx-vlm's prompt and do not say so
    for k, v in setup.items():   # a resumed run must be the same run; older files carry only the model
        if k in res and res[k] != v:
            raise SystemExit(f"results/{a.tag}.json was started with {k}={res[k]!r}, not {v!r}: use another tag")
    for m in meta:
        if m["id"] in res["pages"]:
            continue
        stop = None
        try:
            p = reader.read(d / f"{m['id']}.png", a.scale)
        except guard.MemoryStop as e:   # the page itself was read: keep it, then stop
            p, stop = reader.last, e
        p = {k: v for k, v in p.items() if k != "looped"}
        res["pages"][m["id"]] = p
        print(f"{a.tag} {m['id']:28s} {p['s']:6.1f}s ptok={p['prompt_tokens']:5d} gtok={p['gen_tokens']:5d} {p['gen_tps']:5.1f} tok/s", flush=True)
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        if stop:
            raise stop


def cmd_score(a):
    from . import score
    rows = [score.summarize(a.corpus, t, a.upright) for t in a.tags]
    levels = sorted({l for r in rows for l in r["levels"]})
    print(f"corpus = {a.corpus}" + (", upright pages only" if a.upright else ""))
    print(f"{'tag':26s} {'ent%':>6s} {'95% CI':>13s} {'upright':>7s} {'skipped':>8s} {'loops':>5s} | "
          + " ".join(f"L{l}:ent/cer".rjust(13) for l in levels) + f" | {'s/pg':>5s} {'tok/s':>5s} {'peakGB':>6s} {'n':>4s}")
    for r in rows:
        cells = " ".join((f"{r['levels'][l]['recall']:4.0f}%/{r['levels'][l]['cer']:4.1f}%" if l in r["levels"] else "-").rjust(13) for l in levels)
        print(f"{r['tag']:26s} {r['recall']:5.1f}% [{r['ci'][0]:4.1f}, {r['ci'][1]:4.1f}] {r['recall_upright']:6.1f}% "
              f"{len(r['skipped']):4d}/{r['upright']:<3d} {len(r['loops']):5d} | {cells} | {r['s_page']:5.1f} {r['gen_tps']:5.0f} {r['peak_gb']:6.2f} {r['pages']:4d}")
    types = sorted({t for r in rows for t in r["types"]})
    print("\nentity recall by type:\n" + f"{'tag':26s} " + " ".join(f"{t:>9s}" for t in types))
    for r in rows:
        print(f"{r['tag']:26s} " + " ".join(f"{r['types'].get(t, 0):8.0f}%" for t in types))
    print("\nby language, accented words found as written, amounts returned in the other notation (2.100,00 as 2,100.00):")
    print(f"{'tag':26s} " + " ".join(f"{l:>6s}" for l in sorted({l for r in rows for l in r['langs']})) + f" {'accents':>14s} {'flipped':>12s}")
    for r in rows:
        print(f"{r['tag']:26s} " + " ".join(f"{r['langs'].get(l, 0):5.1f}%" for l in sorted({l for r in rows for l in r['langs']}))
              + f" {r['accent_recall']:5.1f}% of {r['accent_words']:<5d} {r['notation_flips']:4d}/{r['notation_candidates']}")
    if a.pages:
        for r in rows:
            print(f"\n{r['tag']}\n  skipped block: {', '.join(r['skipped']) or '-'}\n  loops: {', '.join(r['loops']) or '-'}")


def cmd_compare(a):
    from . import score
    print(f"B - A, A = {a.base}, corpus = {a.corpus}; verdict by rule 4 (margin -1.5)")
    print(f"{'B':26s} {'A':>6s} {'B':>6s} {'diff':>6s} {'95% CI':>15s}  {'verdict':10s} {'n':>4s}  pages lost/gained, worst")
    for t in a.tags:
        c = score.compare(a.corpus, a.base, t, a.upright)
        print(f"{t:26s} {c['recall_a']:5.1f}% {c['recall_b']:5.1f}% {c['diff']:+5.1f}  [{c['ci'][0]:+5.1f}, {c['ci'][1]:+5.1f}]  "
              f"{score.verdict(c['ci']):10s} {c['pages']:4d}  {len(c['lost'])}/{c['gained']}  " + ", ".join(f"{i} {n}" for n, i in c["lost"][:4]))


def cmd_adaptive(a):
    from . import score
    r = score.adaptive(a.corpus, a.tags, a.upright)
    print(f"corpus {a.corpus}, {r['pages']} pages, {r['entities']} entities, looped in {a.tags[0]}: {r['base_loops']}")
    for row in r["rows"]:
        print(f"{row['policy']:36s} {row['recall']:5.1f}%  {row['s_page']:5.1f} s/page")


def cmd_smoke(a):
    """A new page set in five minutes, before hours are spent generating or reading the rest of it:
    the static check on the whole set, then a few pages per level read with the fast model. Never reads a test set."""
    from . import check, score
    from .engine import Reader, environment
    name = corpus.split_name(a.corpus)[0]
    counts, bad = check.check(name)
    print(f"{name}: {counts['pages']} pages, {counts['entities']} entities, lang {counts['lang']}, level {counts['level']}")
    for kind, ids in bad.items():
        print(f"  PROBLEM {kind}: {len(ids)} ({', '.join(ids[:3])}{', ...' if len(ids) > 3 else ''})")
    if corpus.role(name) == "test":
        print("  test set: static check only, no page is read")
        return 1 if bad else 0
    sample = f"{name}#{a.pages}"
    meta = corpus.load_meta(sample)
    d = corpus.corpus_dir(name)
    tag = "smoke_" + name.strip("/").replace("/", "_")
    out = corpus.result_path(tag)
    out.parent.mkdir(exist_ok=True)
    # a regenerated set keeps its page ids: reads are kept only while the image they were made on is unchanged
    stamp = lambda i: f"{(d / f'{i}.png').stat().st_size}:{(d / f'{i}.png').stat().st_mtime_ns}"
    res = json.load(open(out)) if out.exists() else {}
    pages = {i: p for i, p in res.get("pages", {}).items() if res.get("model") == a.model and res.get("scale") == a.scale
             and p.get("stamp") == (stamp(i) if (d / f"{i}.png").exists() else None)}
    res = {"tag": tag, "model": a.model, "corpus": name, "scale": a.scale, "max_tokens": metrics.MAX_TOKENS, "env": environment(), "pages": pages}
    todo = [m for m in meta if m["id"] not in pages]
    if todo:
        reader = Reader(a.model, mem_gb=a.mem_gb, wait=True, prompt_style=a.prompt)   # queues behind a running job
        for m in todo:
            p = {k: v for k, v in reader.read(d / f"{m['id']}.png", a.scale).items() if k != "looped"}
            res["pages"][m["id"]] = {**p, "stamp": stamp(m["id"])}
            json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
            print(f"  read {m['id']:40s} {p['s']:5.1f}s", flush=True)
    else:
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    r = score.summarize(sample, tag)
    lines = lambda i: [l for l in corpus.reference(name, i).split("\n") if l.strip()]
    text = lambda i: metrics.squash(res["pages"][i]["text"])
    top = sum(metrics.squash(lines(m["id"])[0]) not in text(m["id"]) for m in meta)
    bottom = sum(metrics.squash(lines(m["id"])[-1]) not in text(m["id"]) for m in meta)
    budget = sum(res["pages"][m["id"]]["gen_tokens"] >= metrics.MAX_TOKENS for m in meta)
    print(f"\n{a.model} at {a.scale}x on {r['pages']} pages: entity recall {r['recall']:.1f}% [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}], "
          f"{r['s_page']:.1f} s/page, a full read of the set would take {counts['pages'] * r['s_page'] / 3600:.1f} h")
    print("  by level:    " + "  ".join(f"L{l} {v['recall']:.0f}% (CER {v['cer']:.0f}%)" for l, v in r["levels"].items()))
    print("  by language: " + "  ".join(f"{l} {v:.0f}%" for l, v in r["langs"].items()))
    print("  by type:     " + "  ".join(f"{t} {v:.0f}%" for t, v in r["types"].items()))
    print(f"  pages under 90% of the text {len(r['skipped'])}/{r['upright']}, loops {len(r['loops'])}, token budget hit {budget}, "
          f"first line missing {top}, last line missing {bottom}, amounts in the other notation {r['notation_flips']}/{r['notation_candidates']}")
    warn = []
    first = min(r["levels"])
    if r["levels"][first]["recall"] < 95:
        warn.append(f"clean pages (L{first}) come back at {r['levels'][first]['recall']:.0f}%: look at the ground truth before blaming the model")
    if r["levels"][first]["cer"] > 5:
        warn.append(f"CER of {r['levels'][first]['cer']:.0f}% on clean pages: the .txt and the page do not say the same thing, or not in the same order")
    worst = max(r["levels"])
    if r["levels"][worst]["recall"] < 20:
        warn.append(f"L{worst} comes back at {r['levels'][worst]['recall']:.0f}%: pages nobody can read measure nothing")
    if budget:
        warn.append(f"{budget} pages need more than {metrics.MAX_TOKENS} tokens: too much text on a page for this budget")
    dead = [t for t, v in r["types"].items() if v < 10]
    if dead:
        warn.append(f"entity types never found: {', '.join(dead)}")
    for w in warn:
        print(f"  WARNING {w}")
    print("  verdict: " + ("NOT usable as it is" if bad else "usable, with the warnings above" if warn else "usable"))
    return 1 if bad else 0


def cmd_check(a):
    from . import check
    failed = 0
    for name in a.corpus or list(corpus.registry()):
        try:
            counts, bad = check.check(name)
        except corpus.Missing:
            print(f"{name} [{corpus.registry().get(name, {}).get('role', 'no role')}]: not on this machine")
            failed += bool(a.corpus)   # asked for by name: an error; listed in datasets.toml only: a note
            continue
        print(f"{name} [{corpus.role(name) or 'no role'}]: {counts['pages']} pages, {counts['entities']} entities "
              f"({counts['entities_under_5_chars']} under 5 characters), {counts['templates']} templates, lang {counts['lang']}, level {counts['level']}")
        for kind, ids in bad.items():
            print(f"  PROBLEM {kind}: {len(ids)} ({', '.join(ids[:3])}{', ...' if len(ids) > 3 else ''})")
        failed += bool(bad)
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glmocr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, fn, help):
        p = sub.add_parser(name, help=help, description=fn.__doc__ or help)
        p.set_defaults(fn=fn)
        return p
    add("prepare", cmd_prepare, "download the pinned upstream checkpoint into work/glm-ocr-bf16")
    p = add("quant", cmd_quant, "build a quantized variant, e.g. v8d4e4 or v6d4e4g128, into models/")
    p.add_argument("variant")
    p.add_argument("--src", help="model folder to quantize instead of the original, e.g. one with a fine-tuning adapter fused in")
    p.add_argument("--out", help="destination folder instead of models/VARIANT")
    p = add("infer", cmd_infer, "OCR one page image or a PDF; exit code 2 if an output is a repetition loop")
    p.add_argument("image", help="an image, or a PDF (pages separated by a form feed in the output)")
    p.add_argument("--mode", choices=("auto", "ocr", "text"), default="auto", help="PDF only: auto = the text layer where it passes the checks, the model elsewhere; ocr = always the model; text = always the layer (glmocr/pdf.py)")
    p.add_argument("--pages", help="PDF only: pages to read, counted from 1, e.g. 1-3,7")
    p.add_argument("--model", default=RELEASE, help=f"local model folder or a Hugging Face id (default the release, {RELEASE})")
    p.add_argument("--scale", type=float, default=1.0, help="resize factor before OCR; 1.0 = as is. Below 1 small print is lost (docs/RESULTS.md)")
    p.add_argument("--max-tokens", type=int, default=metrics.MAX_TOKENS)
    p.add_argument("-o", "--output", help="also save a JSON with text and timings")
    p = add("run", cmd_run, "OCR a page set and save results/<tag>.json; resumes a partial run")
    p.add_argument("model")
    p.add_argument("tag")
    p.add_argument("--corpus", default="hard")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=metrics.MAX_TOKENS)
    p.add_argument("--adapter", default=None, help="LoRA adapter folder from a fine-tuning run")
    p.add_argument("--wait", action="store_true", help="queue behind a running model job instead of refusing")
    p = add("score", cmd_score, "entity recall with 95%% CI, skipped blocks, loops, by level and type, speed")
    p.add_argument("tags", nargs="+")
    p.add_argument("--pages", action="store_true", help="also list the pages with a skipped block or a loop")
    p = add("compare", cmd_compare, "paired difference in entity recall of each run against a base run")
    p.add_argument("base")
    p.add_argument("tags", nargs="+")
    p = add("adaptive", cmd_adaptive, "what re-reading a page at another scale would buy")
    p.add_argument("tags", nargs="+", help="base run first, then the same pages at other scales")
    for name in ("infer", "run"):
        sub.choices[name].add_argument("--mem-gb", type=float, default=None, help=f"memory ceiling of the job (default {guard.CEILING_GB:g}, or GLMOCR_MEM_GB)")
    for name in ("score", "compare", "adaptive"):
        sub.choices[name].add_argument("--corpus", default="hard")
        sub.choices[name].add_argument("--upright", action="store_true", help="drop the pages rotated by 90 degrees")
    p = add("smoke", cmd_smoke, "a new page set in five minutes: static check, then a few pages read with the fast model")
    p.add_argument("corpus")
    p.add_argument("--pages", type=int, default=20, help="pages to read, balanced by level (default 20)")
    p.add_argument("--model", default=str(corpus.MODELS / "v8d4e4"))
    p.add_argument("--scale", type=float, default=0.7)
    p.add_argument("--mem-gb", type=float, default=None)
    p = add("check", cmd_check, "validate page sets (default: every set in datasets.toml)")
    p.add_argument("corpus", nargs="*")
    for name, default in (("infer", 1.1), ("run", None)):   # infer reads as release v1.1.0; run measures only what it is told
        sub.choices[name].add_argument("--retry-penalty", type=float, default=default, help=f"read a page that loops a second time with this repetition penalty, 0 for never (default {default}); see engine.keep_retry")
    for name in ("infer", "run", "smoke"):
        sub.choices[name].add_argument("--prompt", choices=["official", "mlx"], default="official", help="official: as the reference implementation; mlx: with /nothink and <think></think>, as every run before 2026-10-14")
    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except guard.MemoryStop as e:
        print(e, file=sys.stderr)
        return guard.STOPPED


if __name__ == "__main__":
    sys.exit(main())
