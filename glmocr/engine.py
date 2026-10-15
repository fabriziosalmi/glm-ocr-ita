"""The one place that calls the model: prompt "Text Recognition:", greedy decoding. Every command reads pages through Reader."""
import pathlib
import tempfile
import time

from . import guard
from .corpus import WORK
from .metrics import MAX_TOKENS, is_loop, loop_start

PROMPT = "Text Recognition:"
TASK_PROMPTS = {"text": PROMPT, "table": "Table Recognition:", "formula": "Formula Recognition:"}   # GLM-OCR's own, per region kind
# mlx-vlm renders the chat template with thinking switched off, which for GLM-OCR appends "/nothink" to the request and
# "<think></think>\n" after "<|assistant|>\n". The reference implementation (transformers) sends neither, and the model
# reads differently: with the official prompt mlx-vlm returns the same text as the reference on 7 pages of 8, among them a
# clean page that came back almost empty with the other. Every run before 2026-10-14 used mlx-vlm's prompt ("mlx").
MLX_EXTRAS = ("/nothink", "<think></think>\n")


def official_prompt(prompt):
    for extra in MLX_EXTRAS:
        prompt = prompt.replace(extra, "")
    return prompt


def environment():
    """What a run was produced with. Outputs are identical byte for byte only within one environment (docs/RESULTS.md, Reproducibility)."""
    import platform
    from importlib.metadata import version
    return {"macos": platform.mac_ver()[0], "python": platform.python_version(),
            **{p: version(p) for p in ("mlx", "mlx-vlm", "transformers", "tokenizers", "pillow", "numpy")}}


class LoopStop:
    """Stopping criterion for generation: the end tokens, or a repetition the text ends with (metrics.loop_start).
    The text is checked every `every` tokens; a page that ends normally is not touched."""
    def __init__(self, tokenizer, eos_ids, every=32):
        self.tok, self.eos, self.every = tokenizer, set(eos_ids), every
        self.ids, self.loop = [], False

    def start(self):
        self.ids, self.loop = [], False
        return self

    def __call__(self, token):
        token = int(token)
        if token in self.eos:
            return True
        self.ids.append(token)
        if len(self.ids) % self.every == 0 and loop_start(self.tok.decode(self.ids)) is not None:
            self.loop = True
        return self.loop


class Reader:
    def __init__(self, model, adapter=None, max_tokens=MAX_TOKENS, mem_gb=None, wait=False, stop_loops=True, prompt_style="official", retry_penalty=None):
        self.ceiling = mem_gb or guard.CEILING_GB
        self._lock = guard.lock(WORK / ".model.lock", wait)
        if not pathlib.Path(model).is_dir():   # a Hugging Face id: download first, so that the fences see the size of the weights
            from huggingface_hub import snapshot_download
            model = snapshot_download(model)
        guard.preflight(model, self.ceiling)
        guard.apply_limits(self.ceiling)
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template
        t = time.time()
        self.model, self.proc = load(str(model), adapter_path=adapter)
        self.load_s = time.time() - t
        assert prompt_style in ("official", "mlx")
        self.prompt_style = prompt_style
        self.prompts = {}
        for task, text in TASK_PROMPTS.items():
            pr = apply_chat_template(self.proc, self.model.config, text, num_images=1)
            if prompt_style == "official":
                pr = official_prompt(pr)
                assert pr.endswith(text + "<|assistant|>\n"), repr(pr[-80:])
            self.prompts[task] = pr
        self.prompt = self.prompts["text"]
        self.max_tokens = max_tokens
        self.retry_penalty = retry_penalty
        eos = self.model.config.eos_token_id
        self.stop = LoopStop(self.proc.tokenizer, eos if isinstance(eos, (list, tuple)) else [eos]) if stop_loops else None
        self._tmp = tempfile.TemporaryDirectory(prefix="glmocr_")

    def read(self, image, scale=1.0, penalty=None, task="text"):
        """image: a path or a PIL image. scale: resize factor per side before OCR (Lanczos); 1.0 hands a file over untouched.
        penalty: a repetition penalty for this one read (mlx-vlm's, over its default context). With retry_penalty set on the
        reader, a page that loops is read a second time with that penalty, and the second reading is kept by keep_retry().
        task: text, table or formula, the prompt GLM-OCR uses for that kind of region (TASK_PROMPTS); a whole page is "text"."""
        import mlx.core as mx
        from PIL import Image
        if isinstance(image, (str, pathlib.Path)) and scale == 1.0:
            path = str(image)
        else:
            im = Image.open(image) if isinstance(image, (str, pathlib.Path)) else image
            if im.mode != "RGB":
                im = im.convert("RGB")
            if scale != 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            path = str(pathlib.Path(self._tmp.name) / "page.png")
            im.save(path)
        mx.reset_peak_memory()
        out = self._generate(path, penalty, task)
        if out["looped"] and penalty is None and self.retry_penalty:
            second = self._generate(path, self.retry_penalty, task)
            first, kept = out, keep_retry(out, second)
            out = dict(second if kept else first)
            out["retry"] = {"penalty": self.retry_penalty, "kept": kept,
                            "first": {k: first[k] for k in ("text", "loop_cut", "s", "gen_tokens", "looped")},
                            "second": {k: second[k] for k in ("text", "loop_cut", "s", "gen_tokens", "looped")}}
            out["s"] = first["s"] + second["s"]
        out["peak_gb"] = mx.get_peak_memory() / 1e9
        self.last = out   # the caller can save it before the watch stops the job
        guard.watch(out["peak_gb"], self.ceiling)
        return out

    def _generate(self, path, penalty, task="text"):
        import mlx.core as mx
        from mlx_vlm import generate
        t = time.time()
        extra = {"stopping_criteria": self.stop.start()} if self.stop else {}
        if penalty:
            extra["repetition_penalty"] = penalty
        r = generate(self.model, self.proc, self.prompts[task], image=path, max_tokens=self.max_tokens, temperature=0.0, verbose=False, **extra)
        text, cut = r.text, None
        if self.stop and self.stop.loop:   # keep what came before the repetition and one unit of it
            hit = loop_start(text)
            if hit:
                cut = len(text) - (hit[0] + hit[1]); text = text[:hit[0] + hit[1]]
        return {"text": text, "loop_cut": cut, "s": time.time() - t, "prompt_tokens": r.prompt_tokens, "gen_tokens": r.generation_tokens,
                "prefill_tps": r.prompt_tps, "gen_tps": r.generation_tps, "peak_gb": mx.get_peak_memory() / 1e9,
                "looped": cut is not None or is_loop(r.text, r.generation_tokens, self.max_tokens)}


def keep_retry(first, second):
    """Which reading of a page that looped to keep, fixed on 2026-10-14 before any retry was measured: the second, read with a
    repetition penalty, if it does not loop itself; if both loop, the one with more text before its loop. No ground truth is used."""
    if not second["looped"]:
        return True
    return len(second["text"]) > len(first["text"])
