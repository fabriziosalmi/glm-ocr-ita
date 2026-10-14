"""python experiments/ft_merge.py ADAPTER_DIR OUT_DIR : the original model with a fine-tuning adapter fused into its weights, in bf16.
Every LoRA pair is added to the weight it adapts, W + scale * B^T A^T, in the file of the original model (work/glm-ocr-bf16);
everything else in the folder is copied as it is. The result is a model folder like the original, which `glmocr quant --src`
then quantizes.
CHECK. The same weights are then computed the library's way: the original loaded with the adapter attached
(mlx_vlm.utils.load(adapter_path=...)) and every LoRA layer fused with its own fuse(). The two must agree on every adapted
weight, or the script fails: a wrong name or a wrong transpose cannot pass silently.
"""
import json, pathlib, shutil, sys
from glmocr import guard
from glmocr.corpus import ROOT, SOURCE, WORK

adapter, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if out.exists(): raise SystemExit(f"{out} exists: choose another folder")
held = guard.lock(WORK / ".model.lock", wait=True)
guard.preflight(SOURCE); guard.apply_limits()
import mlx.core as mx
from mlx.utils import tree_flatten

cfg = json.load(open(adapter / "adapter_config.json"))["lora_parameters"]
scale = cfg["scale"]
lora = mx.load(str(adapter / "adapters.safetensors"))
weights = mx.load(str(SOURCE / "model.safetensors"))
fused = {}
for key in cfg["keys"]:   # language_model.model.layers.N.x  ->  model.language_model.layers.N.x.weight
    name = "model.language_model." + key.removeprefix("language_model.model.") + ".weight"
    w = weights[name]
    a, b = lora[key + ".lora_a"], lora[key + ".lora_b"]   # (in, r), (r, out)
    assert w.shape == (b.shape[1], a.shape[0]), (name, w.shape, a.shape, b.shape)
    weights[name] = fused[name] = (w.astype(mx.float32) + scale * (b.T @ a.T)).astype(w.dtype)
print(f"{len(fused)} weights fused, scale {scale}")

from mlx_vlm.utils import load
model, _ = load(str(SOURCE), adapter_path=str(adapter), processor_config={"trust_remote_code": True})
layers = [(n, m) for n, m in model.named_modules() if hasattr(m, "fuse") and hasattr(m, "lora_a")]
model.update_modules(__import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten([(n, m.fuse()) for n, m in layers]))
lib = dict(tree_flatten(model.parameters()))
worst = 0.0
for name, w in fused.items():
    mine = "language_model.model." + name.removeprefix("model.language_model.")
    d = mx.abs(lib[mine].astype(mx.float32) - w.astype(mx.float32)).max().item()
    worst = max(worst, d)
assert len(layers) == len(fused), (len(layers), len(fused))
assert worst <= 1e-2, f"fused weights differ from the library's by up to {worst}"
print(f"check against the library's fuse on {len(layers)} layers: largest difference {worst:.2e} (bf16 rounding)")
del model, lib

out.mkdir(parents=True)
for f in SOURCE.iterdir():
    if f.is_file() and f.name != "model.safetensors" and not f.name.startswith("."): shutil.copy(f, out / f.name)
mx.save_safetensors(str(out / "model.safetensors"), weights)   # no metadata, as the original file
json.dump({"adapter": str(adapter), "base": str(SOURCE), "fused": len(fused), "scale": scale, "check_max_diff": worst},
          open(out / "merged_from.json", "w"), indent=1)
print(f"-> {out}")
