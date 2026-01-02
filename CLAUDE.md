# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BaSpaCho (Batched Sparse Cholesky) is a high-performance direct solver for symmetric positive-definite sparse matrices. It implements supernodal Cholesky decomposition with CUDA and Metal support for batched GPU solving.

## Build Commands

**Configure (CPU-only, using OpenBLAS):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASPACHO_USE_CUBLAS=0
```

**Configure (with CUDA):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Configure (with Intel MKL):**
```bash
. /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp
```

**Configure (with Apple Metal, macOS only):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBASPACHO_USE_CUBLAS=0 -DBASPACHO_USE_METAL=1 -DBLA_VENDOR=Apple
```

**Build:**
```bash
cmake --build build -- -j16
```

**Run all tests:**
```bash
ctest --test-dir build
```

**List available tests:**
```bash
ctest --test-dir build --show-only
```

**Run a single test:**
```bash
ctest --test-dir build -R <test_name>
```

**Using pixi (alternative):**
```bash
pixi run prepare       # Configure without CUDA
pixi run build         # Build
pixi run test          # Run tests
pixi run build_and_test # Full workflow
```

## Code Style

- C++17 standard
- Google style base with modifications (see `.clang-format`)
- Column limit: 100 characters
- Pointer alignment: left (`int* ptr` not `int *ptr`)
- Format with: `clang-format -i <file>`
- Pre-commit hook runs clang-format automatically

## Architecture

### Core Data Structures

**SparseStructure** (`baspacho/baspacho/SparseStructure.h`): CSR-format sparse structure storing `ptrs` and `inds` vectors representing block indices (not individual elements).

**CoalescedBlockMatrixSkel** (`baspacho/baspacho/CoalescedBlockMatrix.h`): Block matrix skeleton with coalesced columns. Key terminology:
- **span**: basic parameter block grouping
- **lump**: aggregation of consecutive spans
- **chain**: span rows × lump cols
- **board**: all spans in a lump of rows × lump cols

**Solver** (`baspacho/baspacho/Solver.h`): Main interface created via `createSolver()`. Provides:
- `factor()`: Cholesky factorization
- `solve()`, `solveL()`, `solveLt()`: triangular solves
- `factorUpTo()`, `solveLUpTo()`: partial factorization for marginals
- Backends: `BackendRef`, `BackendFast`, `BackendCuda`, `BackendMetal`

### Directory Structure

```
baspacho/
  baspacho/       # Core library sources
  testing/        # Test utilities (TestingMatGen, TestingUtils)
  tests/          # Unit tests (gtest)
  benchmarking/   # Performance benchmarks (bench, BAL_bench)
  examples/       # Example applications (Optimizer, PCG)
```

### Key CMake Options

- `BASPACHO_USE_CUBLAS`: Enable CUDA support (default: ON)
- `BASPACHO_USE_METAL`: Enable Apple Metal support (default: OFF, macOS only, float only)
- `BASPACHO_USE_BLAS`: Enable BLAS support (default: ON)
- `BASPACHO_CUDA_ARCHS`: CUDA architectures ("detect", "torch", or explicit list like "60;70;75")
- `BASPACHO_USE_SUITESPARSE_AMD`: Use SuiteSparse AMD instead of Eigen's implementation
- `BASPACHO_BUILD_TESTS`: Build tests (default: ON)
- `BASPACHO_BUILD_EXAMPLES`: Build examples/benchmarks (default: ON)
- `BLA_VENDOR`: BLAS implementation (ATLAS, OpenBLAS, Intel10_64lp_seq, Apple, etc.)

## GPU Backend Notes

### Metal Backend (Apple Silicon)

The Metal backend provides GPU acceleration on Apple Silicon Macs (M1, M2, M3, etc.).

**Important: Float-only precision.** Apple Silicon GPUs lack native double-precision FP64 support. The Metal backend only supports `float` operations. Attempting to use `double` will result in a clear runtime error.

```cpp
// Metal backend usage (float only)
Settings settings;
settings.backend = BackendMetal;
auto solver = createSolver<float>(paramSize, structure, settings);

// Use MetalMirror for GPU memory management
MetalMirror<float> dataGpu(hostData);
solver.factor(dataGpu.ptr());
dataGpu.get(hostData);  // Copy back to CPU
```

For double precision, use `BackendFast` (CPU with BLAS) or `BackendCuda` (NVIDIA GPU).

### CUDA Backend (NVIDIA)

The CUDA backend supports both float and double precision on NVIDIA GPUs with compute capability >= 6.0.

## Dependencies

Fetched automatically by CMake:
- Eigen 3.4.0
- GoogleTest
- dispenso (multithreading)
- Sophus (for BA examples only)

Optional external:
- CUDA Toolkit (10.2+, architecture >=60 for double atomics)
- CHOLMOD (SuiteSparse) - for benchmarking comparisons
- AMD (SuiteSparse) - alternative reordering algorithm

## Running Benchmarks

```bash
# Compare with CHOLMOD baseline
build/baspacho/benchmarking/bench -B 1_CHOLMOD

# Bundle Adjustment problem
build/baspacho/benchmarking/BAL_bench -i ~/BAL/problem-871-527480-pre.txt

# Collect timing statistics for computation model fitting
build/baspacho/benchmarking/bench -B 1_CHOLMOD -Z
```
