cmake_minimum_required(VERSION 3.14)
project(benchmark_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find Python & pybind11 ────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 via cmake module first, then fall back to python -m pybind11
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pb11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pb11_result
  )
  if (NOT _pb11_result EQUAL 0 OR NOT _pb11_cmakedir)
    message(FATAL_ERROR
      "pybind11 not found. Install it with:\n"
      "    pip install pybind11\n"
      "then re-run cmake.")
  endif()
  # Store in the cache so find_package can pick it up
  set(pybind11_DIR "${_pb11_cmakedir}" CACHE PATH "pybind11 cmake dir" FORCE)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# ── Extension target ──────────────────────────────────────────────────────────
pybind11_add_module(benchmark_core benchmark_core.cpp)

# ── Compiler flags ────────────────────────────────────────────────────────────
if (MSVC)
  target_compile_options(benchmark_core PRIVATE /O2 /arch:AVX2 /W3)
else()
  target_compile_options(benchmark_core PRIVATE
    -O3
    -march=native
    -funroll-loops
    -ffast-math
    -Wall -Wextra -Wno-unused-variable
  )
  # Link pthreads
  find_package(Threads REQUIRED)
  target_link_libraries(benchmark_core PRIVATE Threads::Threads)
endif()

# Output to project root so Python can import directly
set_target_properties(benchmark_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS benchmark_core DESTINATION .)
