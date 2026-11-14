"""python experiments/rivals.py sample                      -> results/rivals_sample.json (100 test pages, fixed seed)
   work/bench_venv/bin/python experiments/rivals.py read NAME  -> results/rival_NAME.json (resumes)
   python experiments/rivals.py score                       -> the table: release v1.5.0 against each rival
Three OCR models of late 2026 against the release on the same 100 test pages (docs/LEDGER.md, "Three other OCR models"). Each rival
reads the whole page with its own plain-text prompt, greedy, at most 3000 new tokens, no retry, as its authors' reference
implementation does where it runs on a Mac: transformers bf16 on the GPU (MPS) for Xiaomi-OCR-0; the official GGUF on llama.cpp for MiniCPM-OCR 1.0,
which transformers does not implement, and for PaddleOCR-VL-1.6, which transformers runs at 2 tokens a second on MPS. The release is its saved v1.5.0 readings: the whole
page with the retry, and the bands where the orientation classifier does not take the page for turned (glmocr infer's default).
Not measured here: the rivals' own layout pipelines (PaddleOCR-VL and MiniCPM-OCR are made to read the regions a layout detector
cuts, as GLM-OCR is), and their speed on the hardware they are made for."""
import argparse, json, pathlib, random, statistics, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from glmocr import corpus, metrics

SETS = {"external/xfund_it_val": ("xfund_it_v8d8e8op1_1.0_op_rp110", 34), "external/ocrbench_test": ("ocrbt_v8d8e8op1_1.0_op_rp110", 33),
        "blind/itdoc_15_correspondence": ("itdoc15_v8d8e8op1_1.0_op_rp110", 33)}
SEED, MAX_NEW = 20261114, 3000
RIVALS = {
    # Read first in transformers 5.23.0 bf16 on MPS (kept as results/rival_xiaomi_ocr0_hf.json): 29 of the 33 ocr-bench pages came back
    # NaN from the first token ("!" repeated); eager attention gave "no discernible text"; float32 does not fit in MPS memory, and on the
    # CPU it reads them right (doc_009) at 10 minutes for 200 tokens. So, as the two others, on llama.cpp: the community BF16 GGUF its
    # authors list (not evaluated by them), thinking off as their Ollama backend renders it, pages fitted within 2048 px as before.
    "xiaomi_ocr0": {"backend": "llama.cpp", "repo": "prithivMLmods/Xiaomi-OCR-0-GGUF", "revision": "7bbfa072e0f3d70ac722e26fbccef5b9f64b7121",
                    "files": ("Xiaomi-OCR-0.BF16.gguf", "Xiaomi-OCR-0.mmproj-bf16.gguf"), "prompt": "Extract the text in the image.",
                    "server": "work/llama-b12377/llama-b12377/llama-server", "template": {"enable_thinking": False}, "fit": 2048},
    # LEDGER, "The harness around MiniCPM-OCR 1.0": glmocr's own llama.cpp reader with MiniCPM-OCR's prompt, and the bands
    "minicpm_ocr10_harness": {"backend": "glmocr", "repo": "openbmb/MiniCPM-OCR-1.0-gguf", "revision": "0133ad8db635210441a55dcf421e25bfb9915142",
                              "files": ("MiniCPM-OCR-1_0-F16.gguf", "mmproj-model-f16-4x.gguf"), "prompt": "Text Recognition:",
                              "server": "work/llama-b12377/llama-b12377/llama-server", "retry_penalty": 1.1},
    "xiaomi_ocr0_hf": {"backend": "transformers", "repo": "SeerRay-Lab/Xiaomi-OCR-0", "revision": "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30",
                    "prompt": "Extract the text in the image.", "chat": {"enable_thinking": False},   # its "text" prompt, evaluation_pipeline
                    "fit": 2048},   # its evaluation fits every image within 2048 x 2048 (evaluation_pipeline/cropping.py, LANCZOS)
    # transformers' PaddleOCR-VL generates about 2 tokens a second on MPS (8.5 s prefill, 32 tokens in 17 s, 2026-11-14): 13 minutes
    # a page. Its official GGUF on llama.cpp instead, as for MiniCPM-OCR.
    "paddleocr_vl16": {"backend": "llama.cpp", "repo": "PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "revision": "511b09642bb324401f15f97cc23bc67e8f0a291d",
                       "files": ("PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"), "prompt": "OCR:",
                       "server": "work/llama-b12377/llama-b12377/llama-server"},
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


def transformers_reader(cfg, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
    model = AutoModelForImageTextToText.from_pretrained(cfg["repo"], revision=cfg["revision"], dtype=getattr(torch, dtype)).to("mps").eval()

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
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # macOS's system proxy would otherwise take 127.0.0.1 too
    srv = subprocess.Popen([str(ROOT / cfg["server"]), "-m", m, "--mmproj", mm, "-c", "16384", "--port", str(port), "-ngl", "99", "--temp", "0"],
                           stdout=subprocess.DEVNULL, stderr=open(ROOT / f"work/rival_llama_{cfg['files'][0].split('.')[0]}.log", "w"))
    for _ in range(300):
        try:
            local.open(f"http://127.0.0.1:{port}/health", timeout=2).read(); break
        except Exception:
            time.sleep(1)
    else:
        srv.kill(); sys.exit("llama-server did not start: see work/rival_llama_*.log")

    def read(image):
        buf = io.BytesIO(); image.save(buf, "PNG")
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": cfg["prompt"]}]}], "temperature": 0, "max_tokens": MAX_NEW, **({"chat_template_kwargs": cfg["template"]} if cfg.get("template") else {})}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(local.open(req, timeout=1800))
        return {"text": r["choices"][0]["message"]["content"], "prompt_tokens": r["usage"]["prompt_tokens"], "gen_tokens": r["usage"]["completion_tokens"]}
    read.server = srv
    return read


NAN = "!" * 20   # Qwen-family models write token 0, "!", once their logits are NaN


def rescue(name):
    """Pages whose reading is NaN from its first token (bf16 on MPS, 2026-11-14) read again in float32; nothing else changes."""
    from PIL import Image
    from glmocr.engine import open_page
    cfg = RIVALS[name]; out = corpus.result_path(f"rival_{name}"); res = json.load(open(out))
    todo = [k for k, r in res["pages"].items() if r["text"].startswith(NAN)]
    print(len(todo), "pages read as NaN"); reader = transformers_reader(cfg, "float32")
    for k in todo:
        c, i = k.split("|"); im = open_page(corpus.corpus_dir(c) / f"{i}.png").convert("RGB")
        if cfg.get("fit"):
            im.thumbnail((cfg["fit"], cfg["fit"]), Image.Resampling.LANCZOS)
        t = time.time(); r = reader(im); r["s"] = time.time() - t; r["dtype"] = "float32"; r["bf16_text"] = res["pages"][k]["text"][:40]
        res["pages"][k] = r; json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"{name} {i:48s} {r['s']:6.1f}s gen={r['gen_tokens']} float32", flush=True)


def glmocr_reader(cfg):
    """glmocr.llama.LlamaReader with MiniCPM-OCR's chat template in place of GLM-OCR's prompt string, wrapped in BandReader."""
    import os
    from huggingface_hub import hf_hub_download
    from glmocr import bands, llama
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"   # macOS's system proxy would take 127.0.0.1 too
    os.environ["GLMOCR_LLAMA_SERVER"] = str(ROOT / cfg["server"])
    m = hf_hub_download(cfg["repo"], cfg["files"][0], revision=cfg["revision"]); hf_hub_download(cfg["repo"], cfg["files"][1], revision=cfg["revision"])
    r = llama.LlamaReader(m, max_tokens=MAX_NEW, retry_penalty=cfg["retry_penalty"], wait=True)
    body = {"messages": [{"role": "user", "content": r.marker + cfg["prompt"]}]}
    req = urllib.request.Request(r._url("/apply-template"), json.dumps(body).encode(), {"Content-Type": "application/json"})
    rendered = json.load(urllib.request.urlopen(req))["prompt"]
    assert rendered.count(r.marker + cfg["prompt"]) == 1, rendered
    llama.PROMPT = rendered.replace(r.marker + cfg["prompt"], "{media}{task}")   # TASK_PROMPTS["text"] is "Text Recognition:" too
    assert llama.TASK_PROMPTS["text"] == cfg["prompt"]
    print("MiniCPM-OCR prompt:", repr(llama.PROMPT), flush=True)
    br = bands.BandReader(r)
    def read(image):
        out = br.read(image)
        return {k: out.get(k) for k in ("text", "whole_text", "bands", "retry", "prompt_tokens", "gen_tokens", "loop_cut")}
    read.server = r.proc
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
    reader = {"transformers": transformers_reader, "llama.cpp": llama_reader, "glmocr": glmocr_reader}[cfg["backend"]](cfg)
    try:
        for p in todo:
            im = open_page(corpus.corpus_dir(p["corpus"]) / f"{p['id']}.png").convert("RGB")
            if cfg.get("fit"):
                im.thumbnail((cfg["fit"], cfg["fit"]), Image.Resampling.LANCZOS)
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


MODEL_ALONE = {"external/xfund_it_val": "xfund_it_v8d8e8op1_1.0_op", "external/ocrbench_test": "ocrbt_v8d8e8op1_1.0_op",
               "blind/itdoc_15_correspondence": "itdoc15_v8d8e8op1_1.0_op_rp110"}   # no sampled letter looped: the retry never ran


def model_alone(keys):
    """The fine-tuned model as the rivals read: the whole page once, no retry, no bands (LEDGER, question 3), on the sampled pages."""
    out = {}
    for name, tag in MODEL_ALONE.items():
        for i, p in corpus.load_run(tag)["pages"].items():
            if f"{name}|{i}" not in keys:
                continue
            assert not p.get("retry"), (tag, i)
            out[f"{name}|{i}"] = {"text": p["text"], "s": p["s"]}
    return out


def score(names, base_name="glm-ocr-ita v1.5.0"):
    pages = json.load(open(corpus.result_path("rivals_sample"))); keys = [f"{p['corpus']}|{p['id']}" for p in pages]
    meta = {f"{n}|{m['id']}": m for n in SETS for m in corpus.load_meta(n)}
    ref = {k: corpus.reference(k.split("|")[0], k.split("|")[1]) for k in keys}
    runs = {"glm-ocr-ita v1.5.0": ours(), "glm-ocr-ita model alone": model_alone(set(keys))}
    for n in names:
        p = corpus.result_path(f"rival_{n}")
        if p.exists():
            runs[n] = json.load(open(p))["pages"]
    groups = {"all 100": keys, **{n.split("/")[1]: [k for k in keys if k.startswith(n)] for n in SETS}}
    count = lambda run, f: {k: (f(meta[k]["entities"], run[k]["text"]), len(meta[k]["entities"])) for k in keys if k in run}
    base = count(runs[base_name], metrics.found); print(f"differences against {base_name}")
    print(f"{'reader':22s} {'pages':>5s} " + " ".join(f"{g:>26s}" for g in groups) + f" {'loose':>6s} {'CER':>6s} {'s/page':>7s}")
    for r, run in runs.items():
        c, cl = count(run, metrics.found), count(run, metrics.found_loose); have = [k for k in keys if k in run]
        cells = []
        for g, ks in groups.items():
            ks = [k for k in ks if k in run]
            if r == base_name or not ks:
                cells.append(f"{metrics.recall(c, ks):6.1f}%".rjust(26) if ks else "-".rjust(26)); continue
            lo, hi = metrics.bootstrap(lambda s: metrics.recall(c, s) - metrics.recall(base, s), ks)
            cells.append(f"{metrics.recall(c, ks):5.1f}% {metrics.recall(c, ks) - metrics.recall(base, ks):+5.1f} [{lo:+5.1f},{hi:+5.1f}]".rjust(26))
        cer = 100 * statistics.mean(metrics.cer(ref[k], run[k]["text"]) for k in have)
        print(f"{r:22s} {len(have):5d} " + " ".join(cells) + f" {metrics.recall(cl, have):5.1f}% {cer:5.1f}% {statistics.mean(run[k]['s'] for k in have):7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "read", "rescue", "score"]); ap.add_argument("name", nargs="?")
    ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    if a.cmd != "sample" and not corpus.ledger_has("rivals_sample"):
        sys.exit("write the question and `rivals_sample` in docs/LEDGER.md first")
    {"sample": sample, "read": lambda: read(a.name, a.limit), "rescue": lambda: rescue(a.name), "score": lambda: [score([n for n in RIVALS if n != "xiaomi_ocr0_hf"], b) for b in ("glm-ocr-ita v1.5.0", "glm-ocr-ita model alone")]}[a.cmd]()
