"""Is a page turned? PP-LCNet_x1_0_doc_ori, PaddleOCR's document orientation classifier (Apache-2.0), in numpy: no PaddleX, no
torch, the same on every platform the reader runs on. It answers 0, 90, 180 or 270, the clockwise turn that straightens the page.

The forward pass follows transformers' pp_lcnet (5.19.0); the preprocessing is PaddleX's (short side to 256 by bilinear interpolation
without antialiasing, centre crop 224, BGR, ImageNet's mean and deviation). With them the classifier gives Phase 1's PaddleX numbers:
159/160 rotations of hard/ and 475/480 of held/ (docs/LEDGER.md, 2026-11-12). experiments/orient_lcnet.py is the torch reference
this module is checked against.

It sees the centre of the page only: on a sparse page whose text sits at the top it can answer a turn with low confidence
(ocrbench_train, 15 of 50 upright pages taken for turned at scores 0.33 to 0.90). So a page counts as turned only when the answer
is not 0 and its score is at least SURE, chosen on the dev sets because the two errors cost differently: an upright page taken
for turned loses its bands (glmocr/bands.py), and with them recall; a turned page banded anyway adds a little CER."""
import json
import struct

import numpy as np

REPO, REVISION = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors", "b2716954d2e321dd4436400b37584fab641027ab"
LABELS = (0, 90, 180, 270)
SURE = 0.8
SIZE, CROP = 256, 224
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)   # B, G, R
EPS = 1e-5
# (stride of the depthwise convolution, squeeze-excitation) of each block, from transformers' PPLCNetConfig; kernels from the weights
BLOCKS = [(1, False), (2, False), (1, False), (2, False), (1, False), (2, False)] + [(1, False)] * 5 + [(2, True), (1, True)]

_weights = None


def read_safetensors(path):
    """name -> float32 array. The format: an 8-byte little-endian header length, a JSON header, the raw tensors."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n)); data = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", (k, v["dtype"])
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(data[a:b], np.float32).reshape(v["shape"])
    return out


def weights():
    global _weights
    if _weights is None:
        from huggingface_hub import hf_hub_download
        _weights = read_safetensors(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
    return _weights


def _resize_crop(x):
    """x: H x W x 3 float32 in [0, 1]. Short side to SIZE (torch's bilinear, align_corners False, no antialiasing, as cv2's
    INTER_LINEAR), then the centre CROP x CROP. Only the cropped pixels are computed."""
    h, w = x.shape[:2]; k = SIZE / min(h, w)
    H, W = round(h * k), round(w * k)
    t, l = (H - CROP) // 2, (W - CROP) // 2

    def axis(n_in, n_out, start):
        d = (np.arange(start, start + CROP) + 0.5) * (n_in / n_out) - 0.5
        d = np.maximum(d, 0)
        i0 = np.minimum(np.floor(d).astype(int), n_in - 1); i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (d - i0).astype(np.float32)
    y0, y1, fy = axis(h, H, t); x0, x1, fx = axis(w, W, l)
    top = x[y0][:, x0] * (1 - fx)[None, :, None] + x[y0][:, x1] * fx[None, :, None]
    bot = x[y1][:, x0] * (1 - fx)[None, :, None] + x[y1][:, x1] * fx[None, :, None]
    return top * (1 - fy)[:, None, None] + bot * fy[:, None, None]


def pixels(image):
    """A PIL image -> 3 x 224 x 224, BGR, normalised."""
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    x = _resize_crop(x)[:, :, ::-1]
    return ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def _bn(x, w, p):
    s = w[p + ".weight"] / np.sqrt(w[p + ".running_var"] + EPS)
    return x * s[:, None, None] + (w[p + ".bias"] - w[p + ".running_mean"] * s)[:, None, None]


def _hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6


def _depthwise(x, k, stride):
    """x: C x H x W, k: C x 1 x kh x kw, padding kh // 2."""
    C, H, W = x.shape; kh = k.shape[2]; p = kh // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - kh) // stride + 1, (W + 2 * p - kh) // stride + 1
    out = np.zeros((C, Ho, Wo), np.float32)
    for i in range(kh):
        for j in range(kh):
            out += xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride] * k[:, 0, i, j][:, None, None]
    return out


def _conv(x, k, stride):
    """Full convolution (the stem): x C x H x W, k O x C x kh x kw, padding kh // 2."""
    C, H, W = x.shape; O, _, kh, _ = k.shape; p = kh // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - kh) // stride + 1, (W + 2 * p - kh) // stride + 1
    cols = np.stack([xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride] for i in range(kh) for j in range(kh)], 1)   # C, kh*kw, Ho, Wo
    return np.tensordot(k.reshape(O, C, kh * kh), cols, axes=([1, 2], [0, 1]))


def _layer(x, w, p, stride=1, depthwise=False):
    k = w[p + ".convolution.weight"]
    x = _depthwise(x, k, stride) if depthwise else (_conv(x, k, stride) if k.shape[2] > 1 else np.tensordot(k[:, :, 0, 0], x, axes=1))
    return _hardswish(_bn(x, w, p + ".normalization"))


def logits(x):
    w = weights()
    x = _layer(x, w, "encoder.convolution", stride=2)
    names = sorted({k.rsplit(".layers.", 1)[0] + ".layers." + k.split(".layers.")[1].split(".")[0] for k in w if ".layers." in k},
                   key=lambda s: (int(s.split(".")[2]), int(s.split(".")[4])))
    assert len(names) == len(BLOCKS), names
    for p, (stride, se) in zip(names, BLOCKS):
        x = _layer(x, w, p + ".depthwise_convolution", stride, depthwise=True)
        if se:
            q = p + ".squeeze_excitation_module.convolutions"
            s = x.mean((1, 2))
            s = np.maximum(w[q + ".0.weight"][:, :, 0, 0] @ s + w[q + ".0.bias"], 0)
            s = np.clip((w[q + ".2.weight"][:, :, 0, 0] @ s + w[q + ".2.bias"]) / 6 + 0.5, 0, 1)
            x = x * s[:, None, None]
        x = _layer(x, w, p + ".pointwise_convolution")
    x = _hardswish(w["last_convolution.weight"][:, :, 0, 0] @ x.mean((1, 2))) * (1 - 0.2)   # dropout's scaling at inference, as the port
    return w["head.weight"] @ x + w["head.bias"]


def classify(image):
    """(clockwise degrees that straighten the page, score)."""
    z = logits(pixels(image)).astype(np.float64)
    p = np.exp(z - z.max()); p /= p.sum()
    k = int(p.argmax())
    return LABELS[k], float(p[k])


def turned(image, sure=SURE):
    """The classifier is sure that the page is not upright."""
    deg, score = classify(image)
    return deg != 0 and score >= sure
