"""Memory discipline for a 16 GB machine. Every model job goes through Reader, and Reader goes through here.

Four fences, from the outside in:
  1. one job at a time    an exclusive lock on work/.model.lock: a second job refuses to start instead of killing the first
  2. room before loading  the weights plus headroom must fit both under the ceiling and in the memory the system has available
  3. a ceiling for MLX    memory limit = the ceiling, buffer cache capped, so freed buffers go back to the system
  4. a watch per page     after every page: stop if the job went over the ceiling or the system is nearly out of memory
MLX treats its memory limit as a guideline, not a wall: fence 4 is the one that actually stops a job. A stopped run has saved
every page it read; the same command resumes it.
"""
import fcntl
import os
import pathlib
import subprocess

GB = 1e9
CEILING_GB = float(os.environ.get("GLMOCR_MEM_GB", 6))   # most a job may use; the bf16 model peaks at 3.2 GB
CACHE_GB = 1.5                                            # most MLX may keep in freed buffers
HEADROOM_GB = 1.0                                         # on top of the weights: a full-resolution page (measured: 0.5 to 0.9)
FLOOR_GB = 1.5                                            # system memory that must stay available while a job runs
STOPPED = 3                                               # exit code of a job stopped by the watch


class MemoryStop(SystemExit):
    def __init__(self, why):
        super().__init__(f"stopped for memory: {why}. Pages read so far are saved; the same command resumes")


def available_gb():
    """Memory the kernel counts as available: its own gauge (kern.memorystatus_level, a percentage of RAM), the one
    macOS acts on when memory runs short. It counts what can be reclaimed, which a sum of free pages does not."""
    q = lambda name: int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout)
    return q("kern.memorystatus_level") / 100 * q("hw.memsize") / GB


def swap_used_gb():
    """Swap in use (vm.swapusage). It grows when the machine is short of memory even while "available" still looks fine."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return float(out.split("used =")[1].split("M")[0].strip().replace(",", ".")) / 1024


def weights_gb(model):
    d = pathlib.Path(model)
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / GB if d.is_dir() else 0.0


def needed_gb(model):
    return weights_gb(model) + HEADROOM_GB


def lock(path, wait=False):
    """Hold the one-job lock for the life of this process. Returns the open file: keep a reference to it.
    wait=True queues behind the running job instead of refusing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        f.seek(0)
        raise SystemExit(f"another model job is running (pid {f.read().strip() or '?'}): one at a time on this machine") from None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def preflight(model, ceiling_gb=CEILING_GB, available=None):
    """Refuse before loading rather than fail after. Returns (needed, available) in GB."""
    need = needed_gb(model)
    have = available_gb() if available is None else available
    if need > ceiling_gb:
        raise SystemExit(f"{model} needs about {need:.1f} GB and the ceiling is {ceiling_gb:.1f} GB (--mem-gb or GLMOCR_MEM_GB)")
    if have < need + FLOOR_GB:
        raise SystemExit(f"{model} needs about {need:.1f} GB plus a floor of {FLOOR_GB:.1f} GB; the system has {have:.1f} GB available. Close something first")
    return need, have


def apply_limits(ceiling_gb=CEILING_GB):
    import mlx.core as mx
    mx.set_memory_limit(int(ceiling_gb * GB))
    mx.set_cache_limit(int(min(CACHE_GB, ceiling_gb / 4) * GB))


def watch(peak_gb, ceiling_gb=CEILING_GB, available=None):
    """Call after every page with the page's peak."""
    if peak_gb > ceiling_gb:
        raise MemoryStop(f"the page peaked at {peak_gb:.2f} GB, over the ceiling of {ceiling_gb:.1f} GB")
    have = available_gb() if available is None else available
    if have < FLOOR_GB:
        raise MemoryStop(f"the system has {have:.2f} GB available, under the floor of {FLOOR_GB:.1f} GB")
