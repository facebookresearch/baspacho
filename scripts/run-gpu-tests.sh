#!/bin/bash
# BaSpaCho GPU Test Runner for Cloud Run
# This script is executed inside the Cloud Run container

set -euo pipefail

echo "=== BaSpaCho GPU Test Runner ==="
echo "Date: $(date)"
echo "Commit: ${GITHUB_SHA:-unknown}"
echo ""

# GPU diagnostics
echo "=== GPU Diagnostics ==="
nvidia-smi || echo "nvidia-smi not available"
echo ""

# Clone repository
echo "=== Cloning Repository ==="
REPO_URL="https://github.com/${GITHUB_REPOSITORY:-facebookresearch/baspacho}.git"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
fi

git clone --depth=1 "$REPO_URL" /workspace/baspacho
cd /workspace/baspacho

# Checkout specific commit if provided
if [ -n "${GITHUB_SHA:-}" ]; then
    echo "Fetching commit: $GITHUB_SHA"
    git fetch --depth=1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
fi

echo ""

# Start sccache and show initial stats
echo "=== sccache Stats (before build) ==="
sccache --start-server 2>/dev/null || true
sccache --show-stats || echo "sccache not available"
echo ""

# Configure and build
echo "=== Configuring CMake ==="
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DBASPACHO_USE_CUBLAS=ON \
    -DBASPACHO_USE_METAL=OFF \
    -DBASPACHO_USE_OPENCL=OFF \
    -DBASPACHO_BUILD_TESTS=ON \
    -DBASPACHO_BUILD_EXAMPLES=ON

echo ""
echo "=== Building ==="
cmake --build build -- -j$(nproc)

echo ""
echo "=== sccache Stats (after build) ==="
sccache --show-stats || true

echo ""

# Run tests
echo "=== Running Tests ==="
cd build
ctest --output-on-failure -j$(nproc)
TEST_RESULT=$?

echo ""

# Run benchmarks if tests passed
if [ $TEST_RESULT -eq 0 ]; then
    echo "=== Running Benchmarks ==="

    # Quick benchmark with a few problems
    echo "--- CUDA Backend ---"
    ./baspacho/benchmarking/bench -S "CUDA" -n 3 2>&1 || echo "CUDA benchmark completed with warnings"

    echo ""
    echo "--- BLAS CPU Backend ---"
    ./baspacho/benchmarking/bench -S "BLAS" -n 3 2>&1 || echo "BLAS benchmark completed with warnings"
fi

echo ""
echo "=== GPU Test Complete ==="
exit $TEST_RESULT
