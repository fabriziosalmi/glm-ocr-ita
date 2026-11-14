"""python experiments/rivals.py sample                      -> results/rivals_sample.json (100 test pages, fixed seed)
   work/bench_venv/bin/python experiments/rivals.py read NAME  -> results/rival_NAME.json (resumes)
   python experiments/rivals.py score                       -> the table: release v1.5.0 against each rival
Three OCR models of late 2026 against the release on the same 100 test pages (docs/LEDGER.md, "Three other OCR models"). Each rival
reads the whole page with its own plain-text prompt, greedy, at most 3000 new tokens, no retry, as its authors' reference
implementation does where it runs on a Mac: transformers bf16 on the GPU (MPS) for Xiaomi-OCR-0 and PaddleOCR-VL-1.6; the official
GGUF on llama.cpp for MiniCPM-OCR 1.0, which transformers does not implement. The release is its saved v1.5.0 readings: the whole
page with the retry, and the bands where the orientation classifier does not take the page for turned (glmocr infer's default).
Not measured here: the rivals' own layout pipelines (PaddleOCR-VL and MiniCPM-OCR are made to read the regions a layout detector
cuts, as GLM-OCR is), and their speed on the hardware they are made for."""
import argparse, json, pathlib, random, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from glmocr import corpus, metrics

SETS = {"external/xfund_it_val": ("xfund_it_v8d8e8op1_1.0_op_rp110", 34), "external/ocrbench_test": ("ocrbt_v8d8e8op1_1.0_op_rp110", 33),
        "blind/itdoc_15_correspondence": ("itdoc15_v8d8e8op1_1.0_op_rp110", 33)}
SEED, MAX_NEW = 20261114, 3000
RIVALS = {
    "xiaomi_ocr0": {"backend": "transformers", "repo": "SeerRay-Lab/Xiaomi-OCR-0", "revision": "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30",
                    "prompt": "Extract the text in the image.", "chat": {"enable_thinking": False}},   # its "text" prompt, evaluation_pipeline
    "paddleocr_vl16": {"backend": "transformers", "repo": "PaddlePaddle/PaddleOCR-VL-1.6", "revision": "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd",
                       "prompt": "OCR:", "chat": {}},
    "minicpm_ocr10": {"backend": "llama.cpp", "repo": "openbmb/MiniCPM-OCR-1.0-gguf", "revision": "0133ad8db635210441a55dcf421e25bfb9915142",
                      "files": ("MiniCPM-OCR-1_0-F16.gguf", "mmproj-model-f16-4x.gguf"), "prompt": "Text Recognition:",
                      "server": "work/llama-b12377/llama-b12377/llama-server"},
}


def sample():
    rng = random.Random(SEED); out = []
    for name, (_, n) in SETS.items():
        ids = sorted(m["id"] for m in corpus.load_meta(name))
        out += [{"corpus": name, "id": i} for i in sorted(rng.sample(ids, n))]
    p = corpus.result_path("rivals_sample"); p.write_text(json.dumps(out, indent=0)); print(len(out), "pages ->", p)


def transformers_reader(cfg):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
    model = AutoModelForImageTextToText.from_pretrained(cfg["repo"], revision=cfg["revision"], dtype=torch.bfloat16).to("mps").eval()

    def read(image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": cfg["prompt"]}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **cfg["chat"]).to("mps")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
        new = out[0, inp["input_ids"].shape[1]:]
        return {"text": proc.decode(new, skip_special_tokens=True), "prompt_tokens": int(inp["input_ids"].shape[1]), "gen_tokens": int(len(new))}
    return read


def llama_reader(cfg):
    import base64, io, subprocess, urllib.request
    from huggingface_hub import hf_hub_download
    m, mm = (hf_hub_download(cfg["repo"], f, revision=cfg["revision"]) for f in cfg["files"])
    port = 8131
    srv = subprocess.Popen([str(ROOT / cfg["server"]), "-m", m, "--mmproj", mm, "-c", "16384", "--port", str(port), "-ngl", "99", "--temp", "0"],
                           stdout=subprocess.DEVNULL, stderr=open(ROOT / "work/rival_llama.log", "w"))
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); break
        except Exception:
            time.sleep(1)
    else:
        srv.kill(); sys.exit("llama-server did not start: work/rival_llama.log")

    def read(image):
        buf = io.BytesIO(); image.save(buf, "PNG")
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": cfg["prompt"]}]}], "temperature": 0, "max_tokens": MAX_NEW}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        return {"text": r["choices"][0]["message"]["content"], "prompt_tokens": r["usage"]["prompt_tokens"], "gen_tokens": r["usage"]["completion_tokens"]}
    read.server = srv
    return read


def read(name, limit=None):
    from PIL import Image
    from glmocr.engine import open_page
    cfg = RIVALS[name]; pages = json.load(open(corpus.result_path("rivals_sample")))[:limit]
    out = corpus.result_path(f"rival_{name}")
    res = json.load(open(out)) if out.exists() else {"tag": f"rival_{name}", **{k: v for k, v in cfg.items() if k != "chat"}, "max_new_tokens": MAX_NEW, "pages": {}}
    todo = [p for p in pages if f"{p['corpus']}|{p['id']}" not in res["pages"]]
    if not todo:
        print("nothing to read"); return
    reader = (transformers_reader if cfg["backend"] == "transformers" else llama_reader)(cfg)
    try:
        for p in todo:
            im = open_page(corpus.corpus_dir(p["corpus"]) / f"{p['id']}.png").convert("RGB")
            t = time.time(); r = reader(im); r["s"] = time.time() - t
            res["pages"][f"{p['corpus']}|{p['id']}"] = r
            json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
            print(f"{name} {p['id']:48s} {r['s']:6.1f}s gen={r['gen_tokens']}", flush=True)
    finally:
        if getattr(reader, "server", None):
            reader.server.kill()


def ours():
    """The release's v1.5.0 reading of each sampled page: whole page plus bands unless the classifier is sure it is turned."""
    from PIL import Image
    from glmocr import bands, orient
    out = {}
    for name, (tag, _) in SETS.items():
        base, pcs = corpus.load_run(tag)["pages"], corpus.load_run(tag + "_pieces")["pages"]; d = corpus.corpus_dir(name)
        for i in base:
            turned = orient.turned(Image.open(d / f"{i}.png"))
            out[f"{name}|{i}"] = {"text": base[i]["text"] if turned else bands.merge_bands(base[i]["text"], pcs[i]),
                                  "s": base[i]["s"] + (0 if turned else pcs[i]["top"]["s"] + pcs[i]["bottom"]["s"])}
    return out


def score(names):
    pages = json.load(open(corpus.result_path("rivals_sample"))); keys = [f"{p['corpus']}|{p['id']}" for p in pages]
    meta = {f"{n}|{m['id']}": m for n in SETS for m in corpus.load_meta(n)}
    ref = {k: corpus.reference(k.split("|")[0], k.split("|")[1]) for k in keys}
    runs = {"glm-ocr-ita v1.5.0": ours()}
    for n in names:
        p = corpus.result_path(f"rival_{n}")
        if p.exists():
            runs[n] = json.load(open(p))["pages"]
    groups = {"all 100": keys, **{n.split("/")[1]: [k for k in keys if k.startswith(n)] for n in SETS}}
    count = lambda run, f: {k: (f(meta[k]["entities"], run[k]["text"]), len(meta[k]["entities"])) for k in keys if k in run}
    base = count(runs["glm-ocr-ita v1.5.0"], metrics.found)
    print(f"{'reader':22s} {'pages':>5s} " + " ".join(f"{g:>26s}" for g in groups) + f" {'loose':>6s} {'CER':>6s} {'s/page':>7s}")
    for r, run in runs.items():
        c, cl = count(run, metrics.found), count(run, metrics.found_loose); have = [k for k in keys if k in run]
        cells = []
        for g, ks in groups.items():
            ks = [k for k in ks if k in run]
            if r == "glm-ocr-ita v1.5.0" or not ks:
                cells.append(f"{metrics.recall(c, ks):6.1f}%".rjust(26) if ks else "-".rjust(26)); continue
            lo, hi = metrics.bootstrap(lambda s: metrics.recall(c, s) - metrics.recall(base, s), ks)
            cells.append(f"{metrics.recall(c, ks):5.1f}% {metrics.recall(c, ks) - metrics.recall(base, ks):+5.1f} [{lo:+5.1f},{hi:+5.1f}]".rjust(26))
        cer = 100 * statistics.mean(metrics.cer(ref[k], run[k]["text"]) for k in have)
        print(f"{r:22s} {len(have):5d} " + " ".join(cells) + f" {metrics.recall(cl, have):5.1f}% {cer:5.1f}% {statistics.mean(run[k]['s'] for k in have):7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "read", "score"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    if a.cmd != "sample" and not corpus.ledger_has("rivals_sample"):
        sys.exit("write the question and `rivals_sample` in docs/LEDGER.md first")
    {"sample": sample, "read": lambda: read(a.name, a.limit), "score": lambda: score(list(RIVALS))}[a.cmd]()
