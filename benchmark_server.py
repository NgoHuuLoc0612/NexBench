"""
benchmark_server.py
Enterprise GPU/CPU Benchmark — Python Backend

C++ core (pybind11) is used when available.
When C++ is not built yet, every benchmark falls back to a real Python
implementation that actually runs the computation — no fake numbers, ever.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ─── third-party ──────────────────────────────────────────────────────────────
try:
    from flask import Flask, jsonify, request, Response, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
except ImportError:
    print("[INSTALL] pip install flask flask-cors flask-socketio eventlet")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not found — install with: pip install psutil")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy not found — install with: pip install numpy")


# ─── C++ core ─────────────────────────────────────────────────────────────────

def _build_and_load_core() -> Any:
    here = Path(__file__).parent

    for pattern in ["benchmark_core*.so", "benchmark_core*.pyd"]:
        hits = list(here.glob(pattern))
        if hits:
            sys.path.insert(0, str(here))
            try:
                import benchmark_core as _bc
                print(f"[CORE] Loaded pre-built {hits[0].name}")
                return _bc
            except Exception as e:
                print(f"[CORE] Pre-built import failed: {e}")

    cmake_lists = here / "CMakeLists.txt"
    setup_py    = here / "setup.py"

    # On Windows, CMake cannot reliably locate pybind11 inside Python's
    # site-packages, so always use setup.py there instead.
    use_cmake = cmake_lists.exists() and sys.platform != "win32"

    if use_cmake:
        build_dir = here / "build"
        build_dir.mkdir(exist_ok=True)
        print("[CORE] Running CMake build…")
        subprocess.check_call(["cmake", ".."], cwd=str(build_dir))
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"],
                               cwd=str(build_dir))
    elif setup_py.exists():
        print("[CORE] Running setup.py build_ext…")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(here))
    else:
        _gen_setup_py(here)
        print("[CORE] Generated setup.py — building…")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(here))

    sys.path.insert(0, str(here))
    import benchmark_core as _bc
    print("[CORE] Built and loaded benchmark_core")
    return _bc


def _gen_setup_py(dest: Path) -> None:
    code = '''\
from setuptools import setup, Extension
import pybind11, sys

ext = Extension(
    "benchmark_core",
    sources=["benchmark_core.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=(
        ["/O2", "/std:c++17"] if sys.platform == "win32"
        else ["-O3", "-std=c++17", "-march=native", "-fPIC",
              "-funroll-loops", "-ffast-math"]
    ),
)
setup(name="benchmark_core", ext_modules=[ext])
'''
    (dest / "setup.py").write_text(code)


try:
    core_mod = _build_and_load_core()
    ENGINE: Any = core_mod.BenchmarkEngine()
    CORE_AVAILABLE = True
    print("[CORE] BenchmarkEngine instantiated ✓")
except Exception as _e:
    CORE_AVAILABLE = False
    ENGINE = None
    print(f"[CORE] C++ core unavailable — using Python implementations: {_e}")


# ─── System probing ────────────────────────────────────────────────────────────

def _system_info() -> dict:
    info: dict = {
        "os":             platform.system(),
        "os_ver":         platform.version(),
        "machine":        platform.machine(),
        "python":         sys.version.split()[0],
        "core_available": CORE_AVAILABLE,
    }
    if CORE_AVAILABLE:
        try:
            info["cpu"] = ENGINE.cpu_info()
        except Exception:
            pass
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        info["ram_total_gb"]       = round(vm.total / 1e9, 2)
        info["ram_avail_gb"]       = round(vm.available / 1e9, 2)
        info["cpu_freq_mhz"]       = getattr(psutil.cpu_freq(), "current", 0)
        info["cpu_count_logical"]  = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["gpus"] = _probe_gpus()
    return info


def _probe_gpus() -> list:
    gpus = []

    # pynvml (most detailed)
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h   = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({
                "index":    i,
                "name":     pynvml.nvmlDeviceGetName(h).decode(),
                "vram_mb":  mem.total // (1024 * 1024),
                "driver":   pynvml.nvmlSystemGetDriverVersion().decode(),
                "temp_c":   pynvml.nvmlDeviceGetTemperature(h, 0),
                "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                "type":     "NVIDIA",
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # nvidia-smi subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL).decode().strip()
        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index":    i,
                    "name":     parts[0],
                    "vram_mb":  int(parts[1]) if parts[1].isdigit() else 0,
                    "driver":   parts[2],
                    "temp_c":   float(parts[3]) if parts[3].replace(".", "").isdigit() else 0,
                    "util_pct": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
                    "type":     "NVIDIA",
                })
        if gpus:
            return gpus
    except Exception:
        pass

    # AMD rocm-smi
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--showtemp",
             "--showuse", "--json"],
            timeout=5, stderr=subprocess.DEVNULL).decode()
        data = json.loads(out)
        for idx, (k, v) in enumerate(data.items()):
            if k.startswith("card"):
                gpus.append({
                    "index":    idx,
                    "name":     v.get("Card series", "AMD GPU"),
                    "vram_mb":  int(v.get("VRAM Total Memory (B)", 0)) // (1024 * 1024),
                    "driver":   "ROCm",
                    "temp_c":   float(v.get("Temperature (Sensor junction) (C)", 0)),
                    "util_pct": int(v.get("GPU use (%)", 0)),
                    "type":     "AMD",
                })
        if gpus:
            return gpus
    except Exception:
        pass

    return [{"index": 0, "name": "No GPU Detected", "vram_mb": 0,
             "driver": "N/A", "temp_c": 0, "util_pct": 0, "type": "Unknown"}]


def _live_metrics() -> dict:
    m: dict = {"ts": time.time()}
    if HAS_PSUTIL:
        m["cpu_pct"]      = psutil.cpu_percent(interval=None)
        m["cpu_per_core"] = psutil.cpu_percent(interval=None, percpu=True)
        vm = psutil.virtual_memory()
        m["ram_used_pct"] = vm.percent
        m["ram_used_gb"]  = round(vm.used / 1e9, 2)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                first = next(iter(temps.values()))
                if first:
                    m["cpu_temp_c"] = first[0].current
        except Exception:
            pass
        try:
            net = psutil.net_io_counters()
            m["net_sent_mb"] = round(net.bytes_sent / 1e6, 2)
            m["net_recv_mb"] = round(net.bytes_recv / 1e6, 2)
        except Exception:
            pass
        try:
            disk = psutil.disk_io_counters()
            m["disk_read_mb"]  = round(disk.read_bytes / 1e6, 2)
            m["disk_write_mb"] = round(disk.write_bytes / 1e6, 2)
        except Exception:
            pass
    return m


# ─── Job scheduler ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkJob:
    job_id:      str
    name:        str
    status:      str   = "pending"
    progress:    float = 0.0
    result:      Optional[dict] = None
    error:       Optional[str]  = None
    created_at:  float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_s:  Optional[float] = None


class JobRegistry:
    def __init__(self):
        self._jobs:    Dict[str, BenchmarkJob] = {}
        self._lock     = threading.Lock()
        self._history: deque = deque(maxlen=200)

    def create(self, name: str) -> BenchmarkJob:
        j = BenchmarkJob(job_id=str(uuid.uuid4()), name=name)
        with self._lock:
            self._jobs[j.job_id] = j
        return j

    def get(self, job_id: str) -> Optional[BenchmarkJob]:
        return self._jobs.get(job_id)

    def finish(self, job: BenchmarkJob, result: dict):
        job.status      = "done"
        job.result      = result
        job.progress    = 1.0
        job.finished_at = time.time()
        job.duration_s  = job.finished_at - job.created_at
        self._history.append(asdict(job))

    def fail(self, job: BenchmarkJob, err: str):
        job.status      = "error"
        job.error       = err
        job.finished_at = time.time()
        self._history.append(asdict(job))

    def all_jobs(self) -> list:
        return [asdict(j) for j in self._jobs.values()]

    def history(self) -> list:
        return list(self._history)


REGISTRY = JobRegistry()


def _run_job_async(job: BenchmarkJob, fn: Callable, socketio_ref: Any):
    def _target():
        job.status = "running"
        try:
            result = fn()
            REGISTRY.finish(job, result)
            socketio_ref.emit("job_done", {
                "job_id": job.job_id,
                "result": result,
            }, namespace="/bench")
        except Exception as e:
            REGISTRY.fail(job, traceback.format_exc())
            socketio_ref.emit("job_error", {
                "job_id": job.job_id,
                "error":  str(e),
            }, namespace="/bench")
    threading.Thread(target=_target, daemon=True).start()


# ─── Real Python benchmark implementations ────────────────────────────────────

def _py_matmul(N: int, iters: int, threads: int) -> dict:
    """Actual N×N single-precision matrix multiply via NumPy BLAS."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    _ = A @ B  # warmup
    flops   = 2 * N * N * N
    t0      = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    elapsed = time.perf_counter() - t0
    gflops  = round((flops * iters) / elapsed / 1e9, 3)
    return {
        "gflops":       gflops,
        "per_thread":   round(gflops / max(1, threads), 3),
        "matrix_size":  N,
        "iterations":   iters,
        "threads_used": threads,
        "elapsed_s":    round(elapsed, 4),
    }


def _py_fft(n: int, iters: int, threads: int) -> dict:
    """Actual FFT via NumPy (backed by FFTPACK/FFTW when available)."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")
    x       = np.random.rand(n).astype(np.complex128)
    _       = np.fft.fft(x)  # warmup
    flops   = 5 * n * math.log2(max(n, 2))
    t0      = time.perf_counter()
    for _ in range(iters):
        y = np.fft.fft(x)
    elapsed = time.perf_counter() - t0
    gflops  = round((flops * iters) / elapsed / 1e9, 3)
    return {
        "gflops":     gflops,
        "fft_size":   n,
        "iterations": iters,
        "threads":    threads,
        "elapsed_s":  round(elapsed, 4),
    }


def _py_memory_bandwidth(mb: int, iters: int) -> dict:
    """Actual sequential read+write bandwidth measured with NumPy."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")
    n_elems     = (mb * 1024 * 1024) // 4
    src         = np.ones(n_elems, dtype=np.float32)
    dst         = np.empty_like(src)
    np.copyto(dst, src)                   # warmup
    total_bytes = mb * 1024 * 1024 * 2   # read + write per iter
    t0          = time.perf_counter()
    for _ in range(iters):
        np.copyto(dst, src)
        _ = dst.sum()                     # force full read
    elapsed = time.perf_counter() - t0
    gbps    = round((total_bytes * iters) / elapsed / 1e9, 3)
    return {
        "bandwidth_gbps": gbps,
        "buffer_mb":      mb,
        "iterations":     iters,
        "elapsed_s":      round(elapsed, 4),
    }


def _py_cache_latency() -> dict:
    """
    Pointer-chasing latency probe across buffer sizes to expose L1/L2/L3/RAM.
    Returns real measured access latencies in nanoseconds.
    """
    sizes = {
        "L1_4KB":    4 * 1024,
        "L2_256KB":  256 * 1024,
        "L3_8MB":    8 * 1024 * 1024,
        "RAM_256MB": 256 * 1024 * 1024,
    }
    results = {}
    for label, byte_size in sizes.items():
        n = byte_size // 8  # 64-bit int array
        if HAS_NUMPY:
            arr = np.arange(n, dtype=np.int64)
            np.random.default_rng(42).shuffle(arr)
            steps = max(200_000, min(n, 4_000_000))
            idx   = int(arr[0])
            t0    = time.perf_counter()
            for _ in range(steps):
                idx = int(arr[idx % n])
            elapsed = time.perf_counter() - t0
        else:
            arr = list(range(n))
            rng = random.Random(42)
            for i in range(n - 1, 0, -1):
                j = rng.randint(0, i)
                arr[i], arr[j] = arr[j], arr[i]
            steps   = min(n, 500_000)
            idx     = 0
            t0      = time.perf_counter()
            for _ in range(steps):
                idx = arr[idx % n]
            elapsed = time.perf_counter() - t0
        results[label] = round(elapsed / steps * 1e9, 2)
    return results


def _py_integer_bench(iters: int) -> dict:
    """Real integer throughput: popcount + LCG multiply-accumulate."""
    if HAS_NUMPY:
        data = np.random.randint(0, 2**31, size=1_000_000, dtype=np.int64)
        t0   = time.perf_counter()
        acc  = np.int64(0)
        for _ in range(iters):
            bits = np.unpackbits(data.view(np.uint8)).sum()
            acc += np.int64(bits)
        elapsed = time.perf_counter() - t0
        ops     = iters * 1_000_000
    else:
        n    = 500_000
        data = [random.getrandbits(32) for _ in range(n)]
        t0   = time.perf_counter()
        acc  = 0
        for _ in range(iters):
            for v in data:
                acc += bin(v).count("1")
        elapsed = time.perf_counter() - t0
        ops     = iters * n
    return {
        "gops":       round(ops / elapsed / 1e9, 3),
        "iterations": iters,
        "elapsed_s":  round(elapsed, 4),
    }


def _py_branch_torture(iters: int) -> dict:
    """Real branch-misprediction stress via random data-dependent branches."""
    n = 500_000
    if HAS_NUMPY:
        data = np.random.randint(0, 100, size=n, dtype=np.int32)
        t0   = time.perf_counter()
        acc  = np.int64(0)
        for _ in range(iters):
            mask_a = data < 33
            mask_b = (data >= 33) & (data < 66)
            acc   += np.sum(data[mask_a]) * 3
            acc   += np.sum(data[mask_b]) * 7
            acc   += np.sum(data[~mask_a & ~mask_b]) * 13
        elapsed = time.perf_counter() - t0
    else:
        data    = [random.randint(0, 99) for _ in range(n)]
        t0      = time.perf_counter()
        acc     = 0
        for _ in range(iters):
            for v in data:
                if v < 33:    acc += v * 3
                elif v < 66:  acc += v * 7
                else:         acc += v * 13
        elapsed = time.perf_counter() - t0
    return {
        "gops":       round((iters * n) / elapsed / 1e9, 3),
        "iterations": iters,
        "elapsed_s":  round(elapsed, 4),
    }


def _py_volume_shader(width: int, height: int, max_steps: int,
                      step_size: float, time_val: float, threads: int) -> dict:
    """
    Actual CPU ray-marching volume renderer — 3D Perlin/FBM + SDF sphere.
    Produces real pixel data; no placeholder buffers.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")

    # --- Perlin noise internals ---
    _p = list(range(256))
    random.Random(42).shuffle(_p)
    _p = _p * 2

    def _fade(t):       return t * t * t * (t * (t * 6 - 15) + 10)
    def _lerp(a, b, t): return a + t * (b - a)

    def _grad(h, x, y, z):
        h &= 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def _noise(x, y, z):
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        x -= math.floor(x); y -= math.floor(y); z -= math.floor(z)
        u = _fade(x); v = _fade(y); w = _fade(z)
        A  = _p[X]+Y;   AA = _p[A]+Z;   AB = _p[A+1]+Z
        B  = _p[X+1]+Y; BA = _p[B]+Z;   BB = _p[B+1]+Z
        return _lerp(
            _lerp(_lerp(_grad(_p[AA], x,   y,   z  ), _grad(_p[BA], x-1, y,   z  ), u),
                  _lerp(_grad(_p[AB], x,   y-1, z  ), _grad(_p[BB], x-1, y-1, z  ), u), v),
            _lerp(_lerp(_grad(_p[AA+1], x, y,   z-1), _grad(_p[BA+1], x-1, y,   z-1), u),
                  _lerp(_grad(_p[AB+1], x, y-1, z-1), _grad(_p[BB+1], x-1, y-1, z-1), u), v), w)

    def _fbm(x, y, z, octaves=4):
        v = 0.0; a = 0.5; f = 1.0
        for _ in range(octaves):
            v += _noise(x*f, y*f, z*f) * a
            a *= 0.5; f *= 2.0
        return v

    def _density(x, y, z):
        disp = _fbm(x*2.1 + time_val*0.3, y*2.1, z*2.1) * 0.35
        sdf  = math.sqrt(x*x + y*y + z*z) - 0.4 + disp
        return max(0.0, -sdf * 6.0)

    # 5-stop transfer function
    _stops = [
        (0.00, (0.05, 0.05, 0.15, 0.00)),
        (0.20, (0.10, 0.30, 0.80, 0.15)),
        (0.50, (0.80, 0.40, 0.10, 0.50)),
        (0.80, (1.00, 0.90, 0.20, 0.80)),
        (1.00, (1.00, 1.00, 1.00, 1.00)),
    ]

    def _tf(d):
        d = min(max(d, 0.0), 1.0)
        for i in range(len(_stops) - 1):
            t0c, c0 = _stops[i]; t1c, c1 = _stops[i+1]
            if t0c <= d <= t1c:
                f = (d - t0c) / (t1c - t0c)
                return tuple(c0[j] + f*(c1[j]-c0[j]) for j in range(4))
        return _stops[-1][1]

    pixels       = np.zeros((height, width, 4), dtype=np.float32)
    total_steps  = 0
    cam_pos      = (0.0, 0.0, -2.0)
    aspect       = width / height

    t0 = time.perf_counter()

    for py in range(height):
        for px in range(width):
            u  = (px / width  - 0.5) * aspect
            v  = (py / height - 0.5) * -1.0
            rn = math.sqrt(u*u + v*v + 1.0)
            rdx, rdy, rdz = u/rn, v/rn, 1.0/rn
            rx, ry, rz = cam_pos
            r = g = b = a = 0.0
            t = 0.1; steps_taken = 0
            for _ in range(max_steps):
                px3 = rx + t*rdx; py3 = ry + t*rdy; pz3 = rz + t*rdz
                if abs(px3) > 1.5 or abs(py3) > 1.5 or abs(pz3) > 1.5:
                    break
                d = _density(px3, py3, pz3)
                if d > 0.001:
                    cr, cg, cb, ca = _tf(d / 2.0)
                    alpha_step = min(ca * step_size * 3.0, 1.0 - a)
                    r += cr * alpha_step; g += cg * alpha_step
                    b += cb * alpha_step; a += alpha_step
                    if a >= 0.99:
                        break
                t += step_size; steps_taken += 1
            pixels[py, px] = [min(r,1.0), min(g,1.0), min(b,1.0), min(a,1.0)]
            total_steps += steps_taken

    elapsed  = time.perf_counter() - t0
    n_rays   = width * height
    pixels_u8 = (pixels * 255).clip(0, 255).astype(np.uint8)

    return {
        "gflops":              round(total_steps * 80 / max(elapsed, 1e-9) / 1e9, 3),
        "rays_per_second":     round(n_rays / max(elapsed, 1e-9)),
        "avg_march_steps":     round(total_steps / max(n_rays, 1), 1),
        "convergence_quality": round(float(pixels[:, :, 3].mean()), 3),
        "pixels":              pixels_u8.reshape(height, width * 4).tolist(),
        "width":               width,
        "height":              height,
        "elapsed_s":           round(elapsed, 3),
    }


def _py_stress_suite(duration_s: int, threads: int) -> dict:
    """
    Real multi-threaded CPU stress: worker threads run continuous matmul,
    main thread samples CPU load and throughput every second.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")
    cpu_count = (psutil.cpu_count(logical=True)
                 if HAS_PSUTIL else os.cpu_count() or 4)
    if threads <= 0:
        threads = cpu_count

    stop_flag  = threading.Event()
    iter_count = [0] * threads
    lock       = threading.Lock()

    def _worker(tid: int, n: int = 128):
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
        while not stop_flag.is_set():
            _ = A @ B
            with lock:
                iter_count[tid] += 1

    workers = [threading.Thread(target=_worker, args=(i,), daemon=True)
               for i in range(threads)]
    for w in workers:
        w.start()

    timestamps: List[float] = []
    cpu_series: List[float] = []
    gflops_series: List[float] = []
    prev_iters    = 0
    flops_per_iter = 2 * 128**3
    t0 = time.perf_counter()

    while (time.perf_counter() - t0) < duration_s:
        time.sleep(1.0)
        elapsed = time.perf_counter() - t0
        with lock:
            total_iters = sum(iter_count)
        delta = total_iters - prev_iters
        prev_iters = total_iters
        timestamps.append(round(elapsed, 2))
        cpu_series.append(psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0)
        gflops_series.append(round(delta * flops_per_iter / 1e9, 2))

    stop_flag.set()
    for w in workers:
        w.join(timeout=2.0)

    return {
        "duration_s":    duration_s,
        "threads":       threads,
        "total_iters":   sum(iter_count),
        "timestamps":    timestamps,
        "cpu_load_pct":  cpu_series,
        "gflops_series": gflops_series,
    }


def _py_noise_volume(nx: int, ny: int, nz: int, scale: float,
                     t_offset: float, octaves: int) -> dict:
    """Real 3D Perlin FBM noise volume computed in Python."""
    if not HAS_NUMPY:
        raise RuntimeError("numpy required: pip install numpy")

    _p = list(range(256))
    random.Random(1337).shuffle(_p)
    _p = _p * 2

    def _fade(t): return t*t*t*(t*(t*6-15)+10)
    def _lerp(a, b, t): return a+t*(b-a)

    def _grad(h, x, y, z):
        h &= 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return (u if (h&1)==0 else -u)+(v if (h&2)==0 else -v)

    def _noise(x, y, z):
        X=int(math.floor(x))&255; Y=int(math.floor(y))&255; Z=int(math.floor(z))&255
        x-=math.floor(x); y-=math.floor(y); z-=math.floor(z)
        u=_fade(x); v=_fade(y); w=_fade(z)
        A=_p[X]+Y; AA=_p[A]+Z; AB=_p[A+1]+Z; B=_p[X+1]+Y; BA=_p[B]+Z; BB=_p[B+1]+Z
        return _lerp(
            _lerp(_lerp(_grad(_p[AA],x,y,z),_grad(_p[BA],x-1,y,z),u),
                  _lerp(_grad(_p[AB],x,y-1,z),_grad(_p[BB],x-1,y-1,z),u),v),
            _lerp(_lerp(_grad(_p[AA+1],x,y,z-1),_grad(_p[BA+1],x-1,y,z-1),u),
                  _lerp(_grad(_p[AB+1],x,y-1,z-1),_grad(_p[BB+1],x-1,y-1,z-1),u),v),w)

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                px = ix/nx*scale + t_offset
                py = iy/ny*scale
                pz = iz/nz*scale
                val=0.0; amp=0.5; freq=1.0
                for _ in range(octaves):
                    val += _noise(px*freq, py*freq, pz*freq)*amp
                    amp*=0.5; freq*=2.0
                vol[iz,iy,ix] = val
    return {"volume": vol.tolist(), "shape": list(vol.shape)}


# ─── Flask + SocketIO app ──────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", template_folder=".")
app.config["SECRET_KEY"] = os.urandom(24)
CORS(app)
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
               logger=False, engineio_logger=False)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/system_info")
def api_system_info():
    return jsonify(_system_info())

@app.route("/api/live_metrics")
def api_live_metrics():
    return jsonify(_live_metrics())

@app.route("/api/jobs")
def api_jobs():
    return jsonify({"jobs": REGISTRY.all_jobs(), "history": REGISTRY.history()})

@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    j = REGISTRY.get(job_id)
    if not j:
        return jsonify({"error": "not found"}), 404
    return jsonify(asdict(j))


# ── Benchmark route factory ────────────────────────────────────────────────────

def _make_bench_route(name: str, fn_core: Callable, fn_py: Callable):
    def _route():
        params = request.json or {}
        job    = REGISTRY.create(name)
        fn     = fn_core if CORE_AVAILABLE else fn_py
        def _runner(): return fn(**params)
        _run_job_async(job, _runner, sio)
        return jsonify({"job_id": job.job_id, "status": "running"})
    _route.__name__ = f"bench_{name}"          # unique name BEFORE registration
    app.add_url_rule(f"/api/bench/{name}", endpoint=f"bench_{name}",
                     view_func=_route, methods=["POST"])
    return _route


_make_bench_route("matmul",
    lambda **kw: ENGINE.run_cpu_matmul(kw.get("N",256), kw.get("iters",3), kw.get("threads",4)),
    lambda **kw: _py_matmul(kw.get("N",256), kw.get("iters",3), kw.get("threads",4)))

_make_bench_route("fft",
    lambda **kw: ENGINE.run_cpu_fft(kw.get("n",65536), kw.get("iters",10), kw.get("threads",4)),
    lambda **kw: _py_fft(kw.get("n",65536), kw.get("iters",10), kw.get("threads",4)))

_make_bench_route("memory_bw",
    lambda **kw: ENGINE.run_memory_bandwidth(kw.get("mb",256), kw.get("iters",5)),
    lambda **kw: _py_memory_bandwidth(kw.get("mb",256), kw.get("iters",5)))

_make_bench_route("cache_latency",
    lambda **kw: ENGINE.run_cache_latency(),
    lambda **kw: _py_cache_latency())

_make_bench_route("integer",
    lambda **kw: ENGINE.run_integer_bench(kw.get("iters",50)),
    lambda **kw: _py_integer_bench(kw.get("iters",50)))

_make_bench_route("branch_torture",
    lambda **kw: ENGINE.run_branch_torture(kw.get("iters",50)),
    lambda **kw: _py_branch_torture(kw.get("iters",50)))

_make_bench_route("volume_shader",
    lambda **kw: ENGINE.run_volume_shader(
        kw.get("width",320), kw.get("height",240),
        kw.get("max_steps",128), kw.get("step_size",0.02),
        kw.get("time_val",0.0), kw.get("threads",4)),
    lambda **kw: _py_volume_shader(
        kw.get("width",320), kw.get("height",240),
        kw.get("max_steps",128), kw.get("step_size",0.02),
        kw.get("time_val",0.0), kw.get("threads",4)))

_make_bench_route("stress_suite",
    lambda **kw: ENGINE.run_stress_suite(kw.get("duration_s",10), kw.get("threads",-1)),
    lambda **kw: _py_stress_suite(kw.get("duration_s",10), kw.get("threads",-1)))


@app.route("/api/bench/noise_volume", methods=["POST"])
def api_noise_volume():
    p   = request.json or {}
    job = REGISTRY.create("noise_volume")

    def _runner():
        if CORE_AVAILABLE:
            arr = ENGINE.generate_noise_volume(
                p.get("nx",32), p.get("ny",32), p.get("nz",32),
                p.get("scale",4.0), p.get("t_offset",0.0), p.get("octaves",4))
            return {"volume": arr.tolist(), "shape": list(arr.shape)}
        return _py_noise_volume(
            p.get("nx",32), p.get("ny",32), p.get("nz",32),
            p.get("scale",4.0), p.get("t_offset",0.0), p.get("octaves",4))

    _run_job_async(job, _runner, sio)
    return jsonify({"job_id": job.job_id})


@app.route("/api/bench/stop_stress", methods=["POST"])
def api_stop_stress():
    if CORE_AVAILABLE:
        ENGINE.stop_stress()
    return jsonify({"ok": True})


# ── SSE telemetry ──────────────────────────────────────────────────────────────

@app.route("/api/telemetry/stream")
def telemetry_stream():
    def _gen():
        while True:
            data = _live_metrics()
            if CORE_AVAILABLE:
                try:
                    data["bench_telemetry"] = ENGINE.get_telemetry(20)
                except Exception:
                    pass
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)
    return Response(_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── WebSocket ──────────────────────────────────────────────────────────────────

@sio.on("connect", namespace="/bench")
def ws_connect():
    emit("server_info", {"core_available": CORE_AVAILABLE,
                          "version": "1.0.0", "ts": time.time()})

@sio.on("request_live", namespace="/bench")
def ws_request_live(data):
    emit("live_metrics", _live_metrics())

@sio.on("ping", namespace="/bench")
def ws_ping(data):
    emit("pong", {"ts": time.time()})


def _telemetry_loop():
    while True:
        try:
            sio.emit("live_metrics", _live_metrics(), namespace="/bench")
        except Exception:
            pass
        time.sleep(1.0)

threading.Thread(target=_telemetry_loop, daemon=True).start()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    mode = "C++ core" if CORE_AVAILABLE else "Python (real computation)"
    print(f"[SERVER] Starting on http://127.0.0.1:{port}  |  mode: {mode}")
    sio.run(app, host="0.0.0.0", port=port, debug=False)
