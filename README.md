# NEXBENCH — CPU/GPU Benchmark Suite

A system benchmarking tool built on a C++17/pybind11 engine with a Python Flask/SocketIO backend and browser frontend. All benchmarks run real computation — no synthetic or fake numbers.

---

## Architecture

```
benchmark_core.cpp   ← C++17 engine (pybind11)
benchmark_server.py  ← Flask + SocketIO backend
index.html / app.js  ← Browser frontend
```

The server auto-builds the C++ extension on first launch. If the build fails, every benchmark falls back to a pure-Python implementation that performs the same real computation.

---

## Requirements

**Python 3.8+**

```
pip install flask flask-cors flask-socketio eventlet psutil numpy pybind11
```

**Optional (GPU detection)**

```
pip install pynvml   # NVIDIA detailed telemetry
```

**Build tools** — one of:

- **Windows**: MSVC or MinGW64 (MSYS2 recommended), CMake ≥ 3.14
- **Linux/macOS**: GCC or Clang, CMake ≥ 3.14

---

## Building the C++ Core

### Windows (recommended — setup.py)

```bat
python setup.py build_ext --inplace
```

### Linux / macOS (CMake)

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The compiled `.pyd` (Windows) or `.so` (Linux/macOS) is placed in the project root and picked up automatically by the server.

### Pre-built binary

A pre-built `benchmark_core_cp311-win_amd64.pyd` for CPython 3.11 / Windows x64 is included. Copy or rename it to `benchmark_core.pyd` and skip the build step.

---

## Running

```bash
python benchmark_server.py
```

Open **http://localhost:5000** in a browser.

Set a custom port via environment variable:

```bash
PORT=8080 python benchmark_server.py
```

---

## Benchmark Suite

| Endpoint | Description | Key metric |
|---|---|---|
| `POST /api/bench/matmul` | N×N F32 matrix multiply (multi-threaded) | GFLOPS |
| `POST /api/bench/fft` | Radix-2 FFT on complex doubles | GFLOPS |
| `POST /api/bench/memory_bw` | Sequential read + write bandwidth | GB/s |
| `POST /api/bench/cache_latency` | Pointer-chasing across L1/L2/L3/RAM | ns per level |
| `POST /api/bench/integer` | Popcount + LCG integer throughput | Gops |
| `POST /api/bench/branch_torture` | Branch misprediction / IPC stress | Gops |
| `POST /api/bench/volume_shader` | Software ray-marching + FBM noise + SDF | GFLOPS + rays/s |
| `POST /api/bench/stress_suite` | Full multi-core stress with telemetry | time-series |
| `POST /api/bench/noise_volume` | 3D FBM Perlin noise volume (WebGL transfer) | — |
| `POST /api/bench/stop_stress` | Abort a running stress suite | — |

All benchmark routes accept a JSON body with optional parameters and return a `job_id` immediately. Results are delivered over WebSocket (`/bench` namespace) when the job completes.

### Example — matrix multiply

```bash
curl -X POST http://localhost:5000/api/bench/matmul \
     -H "Content-Type: application/json" \
     -d '{"N": 512, "iters": 5, "threads": 8}'
```

---

## REST API

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/system_info` | OS, CPU brand/flags, RAM, GPU list |
| `GET` | `/api/live_metrics` | Real-time CPU%, RAM, per-core load |
| `GET` | `/api/jobs` | All active jobs + history |
| `GET` | `/api/jobs/<job_id>` | Single job status / result |
| `GET` | `/api/telemetry/stream` | SSE stream at 2 Hz |

### WebSocket — `/bench` namespace

| Event (client → server) | Event (server → client) | Description |
|---|---|---|
| `connect` | `server_info` | Handshake, reports core availability |
| `request_live` | `live_metrics` | On-demand metrics snapshot |
| `ping` | `pong` | Round-trip latency check |
| — | `live_metrics` | Pushed every 1 s automatically |
| — | `job_done` | Benchmark result on completion |
| — | `job_error` | Error payload if benchmark fails |

---

## C++ Engine Details

`BenchmarkEngine` exposes the following methods via pybind11:

```python
from benchmark_core import BenchmarkEngine
eng = BenchmarkEngine()

eng.cpu_info()                          # vendor, brand, AVX2/AVX-512/FMA/AES-NI flags
eng.run_cpu_matmul(N, iters, threads)
eng.run_cpu_fft(n, iters, threads)
eng.run_memory_bandwidth(mb, iters)
eng.run_cache_latency()
eng.run_integer_bench(iters)
eng.run_branch_torture(iters)
eng.run_volume_shader(width, height, max_steps, step_size, time_val, threads)
eng.run_stress_suite(duration_s, threads)   # threads=-1 → all logical cores
eng.stop_stress()
eng.get_telemetry(last_n)               # ring-buffer snapshot
eng.latency_histogram_stats()           # p50/p95/p99
eng.generate_noise_volume(nx, ny, nz, scale, t_offset, octaves)  # → numpy array
```

**Internal features:**

- 4096-sample telemetry ring-buffer (lock-protected)
- Latency histogram with 256 log-scale buckets and percentile statistics
- CPUID-based feature detection (SSE2, SSE4.1, AVX, AVX2, AVX-512F, FMA, BMI2, AES-NI)
- RDTSC-based high-resolution timing
- Thread pool with `std::future`-based task submission
- Compiler flags: `/O2 /arch:AVX2` (MSVC) or `-O3 -march=native -ffast-math` (GCC/Clang)

---

## GPU Detection

GPU information is collected at startup via (in priority order):

1. **pynvml** — full NVIDIA telemetry (VRAM, temperature, utilization, driver version)
2. **nvidia-smi** subprocess fallback
3. **rocm-smi** subprocess (AMD)

Results are included in `/api/system_info` under the `gpus` key.

---

## Project Structure

```
.
├── benchmark_core.cpp          # C++ engine source
├── benchmark_server.py         # Flask/SocketIO server + Python fallbacks
├── index.html                  # Frontend entry point
├── app.js                      # Frontend logic
├── style.css                   # Frontend styles
├── CMakeLists.txt              # CMake build (Linux/macOS)
├── setup.py                    # setuptools build (Windows / fallback)
└── benchmark_core*.pyd/.so     # Compiled extension (generated)
```
