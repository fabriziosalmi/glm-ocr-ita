"""The source checkpoint and the quantized variants built from it. Nothing is trained here."""
import hashlib
import json
import pathlib
import re
import shutil

from .corpus import MODELS, SOURCE, WORK

RELEASE = "fabriziosalmi/GLM-OCR-ita-mlx-8bit"   # v8d8e8 with the op1 adapter fused in; the same weights since v1.0.0 (docs/RESULTS.md, docs/MODEL_CARD.md)

REPO = "zai-org/GLM-OCR"
REVISION = "2e85a62840ccac27daa451df36c736c4636b8628"
SHA256 = "a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815"   # model.safetensors, 2650 MB
REJECTED_KEYS = ("rope_parameters", "attention_dropout", "in_channels")       # vision_config keys mlx-vlm 0.7.6 refuses


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    """Fetch the pinned revision into work/glm-ocr-bf16 and make its config loadable by mlx-vlm.
    Three keys of vision_config are removed, nothing else changes; the untouched config is kept as work/config.orig.json."""
    from huggingface_hub import snapshot_download
    weights = SOURCE / "model.safetensors"
    if not (weights.exists() and sha256(weights) == SHA256):
        snapshot_download(REPO, revision=REVISION, local_dir=SOURCE)
        assert sha256(weights) == SHA256, "model.safetensors does not match the pinned checksum"
    orig = WORK / "config.orig.json"
    if not orig.exists():
        shutil.copy(SOURCE / "config.json", orig)
    cfg = json.loads(orig.read_text())
    for k in REJECTED_KEYS:
        cfg["vision_config"].pop(k, None)
    (SOURCE / "config.json").write_text(json.dumps(cfg, indent=2))
    return SOURCE


def parse_variant(tag):
    """v<bits>d<bits>e<bits>[g<group>]: vision encoder / decoder / embeddings and output head; 16 = left in bf16.
    g is the quantization group size (default 64): every group stores its own scale and bias."""
    m = re.fullmatch(r"v(\d+)d(\d+)e(\d+)(?:g(\d+))?", tag)
    if not m:
        raise SystemExit(f"'{tag}' is not a variant name: expected v<bits>d<bits>e<bits>[g<group>], e.g. v8d4e4 or v6d4e4g128")
    v, d, e, g = (int(x or 64) for x in m.groups())
    for part, bits in (("encoder", v), ("decoder", d), ("embeddings", e)):
        if not (2 <= bits <= 8 or bits == 16):
            raise SystemExit(f"{part}: {bits} bits is not supported (2 to 8, or 16 to leave it in bf16)")
    if d == 16:
        raise SystemExit("a bf16 decoder is not handled: mlx-vlm takes the decoder's bits as the default width")
    return v, d, e, g


def bits_for(path, v, d, e):
    return v if "vision" in path else e if ("embed_tokens" in path or "lm_head" in path) else d


def quantize(tag, dst=None, src=None):
    """src: a model folder in the layout of the original (default the original itself), e.g. one with an adapter fused in."""
    from . import guard
    v, d, e, g = parse_variant(tag)
    held = guard.lock(WORK / ".model.lock")   # the source model is loaded whole: same fences as a read
    src = pathlib.Path(src) if src else SOURCE
    guard.preflight(src)
    guard.apply_limits()
    from mlx_vlm.convert import convert
    dst = pathlib.Path(dst) if dst else MODELS / tag
    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def predicate(path, module):
        bits = bits_for(path, v, d, e)
        return False if bits >= 16 else {"bits": bits, "group_size": g}

    convert(str(src), str(dst), quantize=True, q_bits=d, q_group_size=g, quant_predicate=predicate)
    for f in src.glob("*.jinja"):
        shutil.copy(f, dst)
    return dst, sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1e6
