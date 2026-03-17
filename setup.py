"""
setup.py
Fallback build script for benchmark_core pybind11 extension.
Used when CMake is unavailable.

Usage:
    python setup.py build_ext --inplace
"""

import sys
from pathlib import Path
from setuptools import setup, Extension

try:
    import pybind11
except ImportError:
    print("[ERROR] pybind11 not installed. Run: pip install pybind11")
    sys.exit(1)

HERE = Path(__file__).parent

# ── Platform-specific compiler flags ─────────────────────────────────────────
if sys.platform == "win32":
    extra_compile = ["/O2", "/std:c++17", "/arch:AVX2", "/W3"]
    extra_link    = []
    define_macros = [("NOMINMAX", None), ("WIN32_LEAN_AND_MEAN", None)]
elif sys.platform == "darwin":
    extra_compile = [
        "-O3", "-std=c++17",
        "-march=native", "-funroll-loops",
        "-ffast-math", "-Wall", "-Wextra",
        "-Wno-unused-variable",
    ]
    extra_link    = ["-stdlib=libc++"]
    define_macros = []
else:  # Linux
    extra_compile = [
        "-O3", "-std=c++17",
        "-march=native", "-funroll-loops",
        "-ffast-math", "-Wall", "-Wextra",
        "-Wno-unused-variable", "-pthread",
    ]
    extra_link    = ["-pthread"]
    define_macros = []

ext = Extension(
    name="benchmark_core",
    sources=["benchmark_core.cpp"],
    include_dirs=[
        pybind11.get_include(),
    ],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
    define_macros=define_macros,
    language="c++",
)

setup(
    name="benchmark_core",
    version="2.0.0",
    description="Enterprise GPU/CPU Benchmark Core Engine (C++ pybind11)",
    author="NEXBENCH",
    ext_modules=[ext],
    zip_safe=False,
    python_requires=">=3.8",
)
