"""Memory discipline for a 16 GB machine. Every model job goes through Reader, and Reader goes through here.

Four fences, from the outside in:
  1. one job at a time    an exclusive lock on work/.model.lock: a second job refuses to start instead of killing the first
  2. room before loading  the weights plus headroom must fit both under the ceiling and in the memory the system has available
  3. a ceiling for MLX    memory limit = the ceiling, buffer cache capped, so freed buffers go back to the system
  4. a watch per page     after every page: stop if the job went over the ceiling or the system is nearly out of memory
MLX treats its memory limit as a guideline, not a wall: fence 4 is the one that actually stops a job. A stopped run has saved
every page it read; the same command resumes it.
The GGUF reader runs on Linux and Windows too: there fence 3 is llama.cpp's own (it holds what the model and its context need),
and fences 2 and 4 read the system's gauge for available memory, a container's limit included.
"""
import os
import pathlib
import subprocess
import sys

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
    """Memory the kernel counts as available. macOS: its own gauge (kern.memorystatus_level, a percentage of RAM), the one
    it acts on when memory runs short; it counts what can be reclaimed, which a sum of free pages does not. Linux:
    MemAvailable, and under a cgroup limit (Docker, LXC) what is left of the limit, whichever is less. Windows: ullAvailPhys."""
    if sys.platform == "darwin":
        q = lambda name: int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout)
        return q("kern.memorystatus_level") / 100 * q("hw.memsize") / GB
    if sys.platform == "win32":
        return _windows_available() / GB
    have = linux_available(pathlib.Path("/proc/meminfo").read_text())
    left = cgroup_left(pathlib.Path("/sys/fs/cgroup"))
    return min(have, left if left is not None else have) / GB


def linux_available(meminfo):
    """Bytes of MemAvailable in the text of /proc/meminfo."""
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise SystemExit("no MemAvailable in /proc/meminfo: kernel older than 3.14")


def cgroup_left(root):
    """Bytes a cgroup v2 limit leaves (memory.max minus memory.current), or None without a limit. In a container
    /proc/meminfo is the host's, so the limit is the number that decides whether the job is killed."""
    limit, used = root / "memory.max", root / "memory.current"
    if not (limit.exists() and used.exists()):
        return None
    top = limit.read_text().strip()
    return None if top == "max" else int(top) - int(used.read_text())


def _windows_available():
    import ctypes
    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                       "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    s = Status(); s.dwLength = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError("GlobalMemoryStatusEx failed")
    return s.ullAvailPhys


def rss_gb(pid):
    """Resident memory of another process (the llama-server): ps on macOS, /proc on Linux, the process's working set on Windows."""
    if sys.platform == "win32":
        return _windows_rss(pid) / GB
    status = pathlib.Path(f"/proc/{pid}/status")
    if status.exists():
        line = next((l for l in status.read_text().splitlines() if l.startswith("VmRSS:")), None)
        return int(line.split()[1]) * 1024 / GB if line else 0.0
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 / GB if out else 0.0


def _windows_rss(pid):
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                    "PagefileUsage", "PeakPagefileUsage")]
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return 0
    try:
        c = Counters(); c.cb = ctypes.sizeof(Counters)
        return c.WorkingSetSize if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb) else 0
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def swap_used_gb():
    """Swap in use (vm.swapusage). It grows when the machine is short of memory even while "available" still looks fine."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return float(out.split("used =")[1].split("M")[0].strip().replace(",", ".")) / 1024


def weights_gb(model):
    """An MLX folder: its safetensors. A .gguf: the file and the mmproj-*.gguf beside it (Q8_0 950 MB + 869 MB; the
    server then holds 2.59 GB resident, 2026-10-17)."""
    d = pathlib.Path(model)
    if d.suffix == ".gguf" and d.is_file():
        return sum(f.stat().st_size for f in {d, *d.parent.glob("mmproj-*.gguf")}) / GB
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / GB if d.is_dir() else 0.0


def needed_gb(model):
    return weights_gb(model) + HEADROOM_GB


def _hold(f, wait):
    """An exclusive lock on the open file, released when the process ends; BlockingIOError if it is held and wait is False."""
    if sys.platform != "win32":
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        return
    import msvcrt
    import time
    while True:   # msvcrt locks a byte range: the first byte stands for the whole file
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if not wait:
                raise BlockingIOError from None
            time.sleep(1)


def lock(path, wait=False):
    """Hold the one-job lock for the life of this process. Returns the open file: keep a reference to it.
    wait=True queues behind the running job instead of refusing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        _hold(f, wait)
    except BlockingIOError:
        f.seek(0)
        try:
            who = f.read().strip()
        except OSError:   # Windows: the locked byte cannot be read by another process
            who = ""
        raise SystemExit(f"another model job is running (pid {who or '?'}): one at a time on this machine") from None
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
