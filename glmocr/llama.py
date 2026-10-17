"""The GGUF reader: the same reading as engine.Reader, through llama.cpp's llama-server, started and stopped here.

A model path ending in .gguf selects it (glmocr run|infer). The vision projector is the mmproj-*.gguf next to it. The server
binary is GLMOCR_LLAMA_SERVER, or llama-server on PATH. What is kept from engine.Reader, so that the two are measured alike:
the official prompt written out token by token (llama-server does not add [gMASK]<sop> itself, and without them the page
loops: 2026-10-17), greedy decoding, the loop stop every 32 tokens (metrics.loop_start), the retry with a repetition penalty
over the last 20 tokens (mlx-vlm's default context) and keep_retry. Not the same: the quantization (llama.cpp's Q8_0, blocks
of 32), the image resize (llama.cpp's, capped at 4096 image tokens for this projector), the kernels."""
import atexit
import base64
import io
import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
import urllib.request

from . import guard
from .corpus import WORK
from .engine import TASK_PROMPTS, keep_retry, open_page, turned_on_disk
from .metrics import MAX_TOKENS, is_loop, loop_start

PROMPT = "[gMASK]<sop><|user|>\n{media}{task}<|assistant|>\n"   # engine's official prompt, the image between its markers by mtmd
REPEAT_LAST_N = 20
EVERY = 32


def server_binary():
    b = os.environ.get("GLMOCR_LLAMA_SERVER") or shutil.which("llama-server")
    if not b or not pathlib.Path(b).exists():
        raise SystemExit("llama-server not found: set GLMOCR_LLAMA_SERVER or put it on PATH (llama.cpp)")
    return b


def mmproj_for(model):
    found = sorted(pathlib.Path(model).parent.glob("mmproj-*.gguf"))
    if len(found) != 1:
        raise SystemExit(f"expected one mmproj-*.gguf next to {model}, found {len(found)}")
    return found[0]


class LlamaReader:
    def __init__(self, model, adapter=None, max_tokens=MAX_TOKENS, mem_gb=None, wait=False, stop_loops=True, prompt_style="official", retry_penalty=None, ctx=8192):
        if adapter:
            raise SystemExit("a GGUF has its adapter fused in: drop --adapter")
        if prompt_style != "official":
            raise SystemExit("the GGUF reader writes the official prompt only")
        self.ceiling = mem_gb or guard.CEILING_GB
        self._lock = guard.lock(WORK / ".model.lock", wait)
        self.model, self.max_tokens, self.stop_loops, self.retry_penalty = str(model), max_tokens, stop_loops, retry_penalty
        self.prompt_style = prompt_style
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]
        log = open(WORK / "llama-server.log", "w")
        t = time.time()
        self.proc = subprocess.Popen([server_binary(), "-m", self.model, "--mmproj", str(mmproj_for(model)), "-fa", "on", "-c", str(ctx),
                                      "-np", "1", "--host", "127.0.0.1", "--port", str(self.port)], stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({self.proc.returncode}): see {WORK / 'llama-server.log'}")
            try:
                if json.load(urllib.request.urlopen(self._url("/health"), timeout=2)).get("status") == "ok":
                    break
            except OSError:
                pass
            time.sleep(0.5)
        self.load_s = time.time() - t
        props = json.load(urllib.request.urlopen(self._url("/props")))
        self.marker, self.build_info = props["media_marker"], props.get("build_info", "?")

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait(timeout=30)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def rss_gb(self):
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True).stdout.strip()
        return int(out) * 1024 / 1e9 if out else 0.0

    def read(self, image, scale=1.0, penalty=None, task="text", deskew=None, trim=False):
        """As engine.Reader.read. peak_gb is the server's resident memory after the read, not a GPU peak."""
        from PIL import Image
        skew = None
        if deskew is not None:
            from . import deskew as dk
            im = open_page(image)
            out_im, skew = dk.straighten(im, deskew, trim)
            if out_im is not im:
                image = out_im
        if isinstance(image, (str, pathlib.Path)) and scale == 1.0 and not turned_on_disk(image):
            data = pathlib.Path(image).read_bytes()   # the file as it is; llama.cpp's decoder does not apply an EXIF orientation
        else:
            im = open_page(image).convert("RGB")
            if scale != 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, format="PNG"); data = buf.getvalue()
        out = self._generate(data, penalty, task)
        if out["looped"] and penalty is None and self.retry_penalty:
            second = self._generate(data, self.retry_penalty, task)
            first, kept = out, keep_retry(out, second)
            out = dict(second if kept else first)
            out["retry"] = {"penalty": self.retry_penalty, "kept": kept,
                            "first": {k: first[k] for k in ("text", "loop_cut", "s", "gen_tokens", "looped")},
                            "second": {k: second[k] for k in ("text", "loop_cut", "s", "gen_tokens", "looped")}}
            out["s"] = first["s"] + second["s"]
        out["peak_gb"] = self.rss_gb()
        if skew is not None:
            out["skew_deg"], out["deskewed"] = skew, abs(skew) >= deskew
        self.last = out
        guard.watch(out["peak_gb"], self.ceiling)
        return out

    def _generate(self, data, penalty, task="text"):
        body = {"prompt": {"prompt_string": PROMPT.format(media=self.marker, task=TASK_PROMPTS[task]),
                           "multimodal_data": [base64.b64encode(data).decode()]},
                "n_predict": self.max_tokens, "temperature": 0, "top_k": 1, "cache_prompt": False, "stream": True}
        if penalty:
            body.update(repeat_penalty=penalty, repeat_last_n=REPEAT_LAST_N)
        t = time.time(); first = None; text = ""; n = 0; ptok = 0; timings = {}; stopped = False
        req = urllib.request.Request(self._url("/completion"), json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                d = json.loads(line[5:])
                if first is None:
                    first = time.time()
                text += d.get("content", ""); n = d.get("tokens_predicted", n); ptok = d.get("tokens_evaluated", ptok)
                if d.get("stop"):
                    timings = d.get("timings", {}); break
                if self.stop_loops and n % EVERY == 0 and loop_start(text) is not None:
                    stopped = True; break      # closing the stream makes the server stop this generation
        s = time.time() - t
        cut = None
        if stopped:
            hit = loop_start(text)
            cut = len(text) - (hit[0] + hit[1]); text = text[:hit[0] + hit[1]]
        gen_s = (time.time() - first) if first else 0.0
        return {"text": text, "loop_cut": cut, "s": s, "prompt_tokens": ptok, "gen_tokens": n,
                "prefill_tps": timings.get("prompt_per_second") or (ptok / max((first or t) - t, 1e-9)),
                "gen_tps": timings.get("predicted_per_second") or (n / max(gen_s, 1e-9)),
                "looped": cut is not None or is_loop(text, n, self.max_tokens)}
