"""python experiments/ft_points.py NAME STEP [STEP...] : read the adapters a training run saved along the way on the quick panel.
For each STEP: work/ft/NAME/<STEP>_adapters.safetensors becomes the adapter folder work/ft/NAME_STEP/, the panel is read with the
original model plus that adapter (experiments/panel.py read, results/panel_*_ft_NAME_STEP.json), and scored against the original.
The steps are read in order and the reading stops at the first one the stopping rule of docs/ROADMAP.md discards."""
import shutil, subprocess, sys
from glmocr.corpus import ROOT, SOURCE, WORK
base = "bf16_1.0_op"   # the original model on the panel, official prompt
name, steps = sys.argv[1], [int(x) for x in sys.argv[2:]]
run = WORK / "ft" / name
for step in steps:
    d = WORK / "ft" / f"{name}_{step}"; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / f"{step:07d}_adapters.safetensors", d / "adapters.safetensors"); shutil.copy(run / "adapter_config.json", d)
    tag = f"ft_{name}_{step}"
    for attempt in range(4):   # a Metal GPU timeout ends a read now and then; the same command resumes it
        if subprocess.run([sys.executable, "experiments/panel.py", "read", str(SOURCE), tag, "--adapter", str(d)], cwd=ROOT).returncode == 0: break
    out = subprocess.run([sys.executable, "experiments/panel.py", "score", base, tag], cwd=ROOT, capture_output=True, text=True).stdout
    print(out, flush=True)
    if "stopping rule: STOP" in out or "INCOMPLETE" in out:
        print(f"step {step}: reading stops here"); break
