/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef BASPACHO_USE_OPENCL

#include <chrono>
#include <iostream>
#include <clblast.h>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MathUtils.h"
#include "baspacho/baspacho/OpenCLDefs.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

using OuterStride = Eigen::OuterStride<>;
template <typename T>
using OuterStridedMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatM =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, OuterStride>;
template <typename T>
using OuterStridedCMajMatK =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
               OuterStride>;
template <typename T>
using MatRMaj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Synchronization ops for OpenCL
struct OpenCLSyncOps {
  static void sync() {
    OpenCLContext::instance().synchronize();
  }
};

struct OpenCLSymElimCtx : SymElimCtx {
  OpenCLSymElimCtx() {}
  virtual ~OpenCLSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  OpenCLMirror<int64_t> makeBlockPairEnumStraight;
};

struct OpenCLSymbolicCtx : SymbolicCtx {
  OpenCLSymbolicCtx(const CoalescedBlockMatrixSkel& skel, const std::vector<int64_t>& permutation)
      : skel(skel) {
    devLumpToSpan.load(skel.lumpToSpan);
    devChainRowsTillEnd.load(skel.chainRowsTillEnd);
    devChainRowSpan.load(skel.chainRowSpan);
    devSpanOffsetInLump.load(skel.spanOffsetInLump);
    devLumpStart.load(skel.lumpStart);
    devChainColPtr.load(skel.chainColPtr);
    devChainData.load(skel.chainData);
    devBoardColPtr.load(skel.boardColPtr);
    devBoardChainColOrd.load(skel.boardChainColOrd);
    devSpanStart.load(skel.spanStart);
    devSpanToLump.load(skel.spanToLump);
    devPermutation.load(permutation);
  }

  virtual ~OpenCLSymbolicCtx() override {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    // Note: OpenCL doesn't support device pointers in the same way
    // This would need proper SVM (shared virtual memory) support
    PermutedCoalescedAccessor retv;
    // For now, return empty accessor - device operations use buffers directly
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    OpenCLSymElimCtx* elim = new OpenCLSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t n = endPtr - startPtr;
      makeStraight[l - lumpsBegin] = n * (n + 1) / 2;
    }
    cumSumVec(makeStraight);

    elim->numColumns = lumpsEnd - lumpsBegin;
    elim->numBlockPairs = makeStraight[makeStraight.size() - 1];
    elim->makeBlockPairEnumStraight.load(makeStraight);

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  const CoalescedBlockMatrixSkel& skel;

  OpenCLMirror<int64_t> devLumpToSpan;
  OpenCLMirror<int64_t> devChainRowsTillEnd;
  OpenCLMirror<int64_t> devChainRowSpan;
  OpenCLMirror<int64_t> devSpanOffsetInLump;
  OpenCLMirror<int64_t> devLumpStart;
  OpenCLMirror<int64_t> devChainColPtr;
  OpenCLMirror<int64_t> devChainData;
  OpenCLMirror<int64_t> devBoardColPtr;
  OpenCLMirror<int64_t> devBoardChainColOrd;
  OpenCLMirror<int64_t> devSpanStart;
  OpenCLMirror<int64_t> devSpanToLump;
  OpenCLMirror<int64_t> devPermutation;
};

// OpenCL ops factory
struct OpenCLOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new OpenCLSymbolicCtx(skel, permutation));
  }
};

template <typename T>
struct OpenCLNumericCtx : NumericCtx<T> {
  OpenCLNumericCtx(const OpenCLSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : spanToChainOffset(numSpans), tempBuffer(bufSize), sym(sym) {
    devTempBuffer.resizeToAtLeast(bufSize);
    devSpanToChainOffset.resizeToAtLeast(spanToChainOffset.size());
  }

  virtual ~OpenCLNumericCtx() override {}

  // GPU<->CPU sync helpers for operations that need CPU fallback
  // Downloads a region from GPU to CPU buffer, returns pointer to CPU data
  T* downloadRegion(T* data, int64_t offset, int64_t count) {
    auto [buffer, baseOffset] = OpenCLBufferRegistry::instance().findBuffer(data);
    if (!buffer) {
      // Data is already on CPU
      return data + offset;
    }

    // Ensure CPU cache is large enough
    size_t neededSize = offset + count;
    if (cpuDataCache.size() < neededSize) {
      cpuDataCache.resize(neededSize);
    }

    // Download the region from GPU
    size_t byteOffset = baseOffset + offset * sizeof(T);
    clCHECK(clEnqueueReadBuffer(OpenCLContext::instance().queue(),
                                buffer, CL_TRUE, byteOffset,
                                count * sizeof(T), cpuDataCache.data() + offset,
                                0, nullptr, nullptr));
    return cpuDataCache.data() + offset;
  }

  // Uploads a region from CPU cache back to GPU
  void uploadRegion(T* data, int64_t offset, int64_t count) {
    auto [buffer, baseOffset] = OpenCLBufferRegistry::instance().findBuffer(data);
    if (!buffer) {
      // Data is on CPU, copy from cache if needed
      if (cpuDataCache.data() + offset != data + offset) {
        std::copy(cpuDataCache.data() + offset, cpuDataCache.data() + offset + count, data + offset);
      }
      return;
    }

    // Upload the region to GPU
    size_t byteOffset = baseOffset + offset * sizeof(T);
    clCHECK(clEnqueueWriteBuffer(OpenCLContext::instance().queue(),
                                 buffer, CL_TRUE, byteOffset,
                                 count * sizeof(T), cpuDataCache.data() + offset,
                                 0, nullptr, nullptr));
  }

  // Check if data is on GPU
  bool isOnGpu(const T* data) const {
    auto [buffer, offset] = OpenCLBufferRegistry::instance().findBuffer(data);
    return buffer != nullptr;
  }

  // Get GPU buffer for data, returns {buffer, byteOffset} or {nullptr, 0}
  std::pair<cl_mem, size_t> getGpuBuffer(const T* data) const {
    return OpenCLBufferRegistry::instance().findBuffer(data);
  }

  std::vector<T> cpuDataCache;  // Cache for CPU fallback operations

  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) override {
    // TODO: Implement using OpenCL kernel
    BASPACHO_CHECK(false && "pseudoFactorSpans not yet implemented for OpenCL");
  }

  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const OpenCLSymElimCtx* pElim = dynamic_cast<const OpenCLSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const OpenCLSymElimCtx& elim = *pElim;
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // Try to find GPU buffer for data
    auto [dataBuffer, dataOffset] = OpenCLBufferRegistry::instance().findBuffer(data);

    // Use GPU path when data is on GPU
    // Note: Double precision kernels are not supported on Apple Silicon, so fall back to CPU
    bool useGpuPath = (dataBuffer != nullptr);
    if constexpr (std::is_same<T, double>::value) {
      // Check if double precision kernels are available
      static bool doubleKernelAvailable =
          OpenCLContext::instance().hasKernel("factor_lumps_kernel_double");
      if (!doubleKernelAvailable) {
        useGpuPath = false;
      }
    }

    if (!useGpuPath) {
      // CPU fallback path - either data is on CPU, or GPU kernels unavailable (e.g., double on Apple)
      // If data is on GPU, we need to sync: download, operate on CPU, upload
      T* workData = data;
      bool needSync = (dataBuffer != nullptr);
      if (needSync) {
        // Download entire data array from GPU
        size_t dataSize = skel.dataSize();
        if (cpuDataCache.size() < dataSize) {
          cpuDataCache.resize(dataSize);
        }
        clCHECK(clEnqueueReadBuffer(OpenCLContext::instance().queue(),
                                    dataBuffer, CL_TRUE, dataOffset,
                                    dataSize * sizeof(T), cpuDataCache.data(),
                                    0, nullptr, nullptr));
        workData = cpuDataCache.data();
      }

      // CPU fallback (reference implementation style)
      for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
        // Factor the diagonal lump using CPU
        int64_t lumpStart = skel.lumpStart[l];
        int64_t lumpSize = skel.lumpStart[l + 1] - lumpStart;
        int64_t colStart = skel.chainColPtr[l];
        int64_t diagDataPtr = skel.chainData[colStart];

        // In-place Cholesky on diagonal block
        Eigen::Map<MatRMaj<T>> diagBlock(workData + diagDataPtr, lumpSize, lumpSize);
        Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(diagBlock);

        // Below-diagonal solve
        int64_t colEnd = skel.chainColPtr[l + 1];
        for (int64_t ptr = colStart + 1; ptr < colEnd; ptr++) {
          int64_t rowSpan = skel.chainRowSpan[ptr];
          int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - skel.spanStart[rowSpan];
          int64_t blockPtr = skel.chainData[ptr];
          Eigen::Map<MatRMaj<T>> block(workData + blockPtr, rowSpanSize, lumpSize);
          diagBlock.template triangularView<Eigen::Lower>().solveInPlace(block.transpose());
        }
      }

      // Sparse elimination on CPU using element-by-element accumulation.
      // This matches the reference BLAS implementation (elimBlock in MatOpsCpuBase.h)
      // for numerical precision. Using Eigen's gemm (noalias() -=) produces different
      // rounding due to SIMD/blocking reordering operations, causing ~10,000x worse
      // precision (1e-4 vs 1e-8) due to floating-point non-associativity.
      for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
        int64_t startPtr = skel.chainColPtr[l] + 1;
        int64_t endPtr = skel.chainColPtr[l + 1];
        int64_t lColSize = skel.lumpStart[l + 1] - skel.lumpStart[l];

        for (int64_t i = startPtr; i < endPtr; i++) {
          int64_t si = skel.chainRowSpan[i];
          int64_t siSize = skel.spanStart[si + 1] - skel.spanStart[si];
          int64_t siDataPtr = skel.chainData[i];
          T* ilData = workData + siDataPtr;  // siSize x lColSize, row-major

          int64_t targetLump = skel.spanToLump[si];
          int64_t targetSpanOffset = skel.spanOffsetInLump[si];
          int64_t targetStartPtr = skel.chainColPtr[targetLump];
          int64_t targetEndPtr = skel.chainColPtr[targetLump + 1];
          int64_t targetLumpSize = skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];

          for (int64_t j = i; j < endPtr; j++) {
            int64_t sj = skel.chainRowSpan[j];
            int64_t sjSize = skel.spanStart[sj + 1] - skel.spanStart[sj];
            int64_t sjDataPtr = skel.chainData[j];
            T* jlData = workData + sjDataPtr;  // sjSize x lColSize, row-major

            uint64_t pos = bisect(skel.chainRowSpan.data() + targetStartPtr,
                                  targetEndPtr - targetStartPtr, sj);
            int64_t jiDataPtr = skel.chainData[targetStartPtr + pos];
            T* jiData = workData + jiDataPtr + targetSpanOffset;  // sjSize x siSize, stride targetLumpSize

            // Element-by-element: jiBlock -= jlBlock * ilBlock.transpose()
            // This is equivalent to: A[r,c] -= sum_k(B[r,k] * C[c,k]) for all r,c
            // where A=jiData (sjSize x siSize), B=jlData (sjSize x lColSize), C=ilData (siSize x lColSize)
            for (int64_t r = 0; r < sjSize; r++) {
              T* jiRow = jiData + r * targetLumpSize;
              T* jlRow = jlData + r * lColSize;
              for (int64_t c = 0; c < siSize; c++) {
                T* ilRow = ilData + c * lColSize;
                T& v = jiRow[c];
                for (int64_t k = 0; k < lColSize; k++) {
                  v -= jlRow[k] * ilRow[k];
                }
              }
            }
          }
        }
      }

      // Upload results back to GPU if needed
      if (needSync) {
        size_t dataSize = skel.dataSize();
        clCHECK(clEnqueueWriteBuffer(OpenCLContext::instance().queue(),
                                     dataBuffer, CL_TRUE, dataOffset,
                                     dataSize * sizeof(T), cpuDataCache.data(),
                                     0, nullptr, nullptr));
      }
      return;
    }

    // GPU path - use OpenCL kernels
    cl_command_queue queue = OpenCLContext::instance().queue();
    int64_t numLumps = lumpsEnd - lumpsBegin;

    // Get buffer handles (need to store in locals to take address)
    cl_mem lumpStartBuf = sym.devLumpStart.buffer();
    cl_mem chainColPtrBuf = sym.devChainColPtr.buffer();
    cl_mem chainDataBuf = sym.devChainData.buffer();
    cl_mem boardColPtrBuf = sym.devBoardColPtr.buffer();
    cl_mem boardChainColOrdBuf = sym.devBoardChainColOrd.buffer();
    cl_mem chainRowsTillEndBuf = sym.devChainRowsTillEnd.buffer();
    cl_mem chainRowSpanBuf = sym.devChainRowSpan.buffer();
    cl_mem spanStartBuf = sym.devSpanStart.buffer();
    cl_mem spanToLumpBuf = sym.devSpanToLump.buffer();
    cl_mem spanOffsetInLumpBuf = sym.devSpanOffsetInLump.buffer();

    // Factor lumps kernel
    const char* factorKernelName = std::is_same<T, float>::value
        ? "factor_lumps_kernel_float" : "factor_lumps_kernel_double";
    cl_kernel factorKernel = OpenCLContext::instance().getKernel(factorKernelName);

    clCHECK(clSetKernelArg(factorKernel, 0, sizeof(cl_mem), &lumpStartBuf));
    clCHECK(clSetKernelArg(factorKernel, 1, sizeof(cl_mem), &chainColPtrBuf));
    clCHECK(clSetKernelArg(factorKernel, 2, sizeof(cl_mem), &chainDataBuf));
    clCHECK(clSetKernelArg(factorKernel, 3, sizeof(cl_mem), &boardColPtrBuf));
    clCHECK(clSetKernelArg(factorKernel, 4, sizeof(cl_mem), &boardChainColOrdBuf));
    clCHECK(clSetKernelArg(factorKernel, 5, sizeof(cl_mem), &chainRowsTillEndBuf));
    clCHECK(clSetKernelArg(factorKernel, 6, sizeof(cl_mem), &dataBuffer));
    clCHECK(clSetKernelArg(factorKernel, 7, sizeof(int64_t), &lumpsBegin));
    clCHECK(clSetKernelArg(factorKernel, 8, sizeof(int64_t), &lumpsEnd));

    size_t globalSize = numLumps;
    clCHECK(clEnqueueNDRangeKernel(queue, factorKernel, 1, nullptr, &globalSize,
                                   nullptr, 0, nullptr, nullptr));

    // Sparse elimination kernel
    if (elim.numBlockPairs > 0) {
      const char* elimKernelName = std::is_same<T, float>::value
          ? "sparse_elim_straight_kernel_float" : "sparse_elim_straight_kernel_double";
      cl_kernel elimKernel = OpenCLContext::instance().getKernel(elimKernelName);
      cl_mem enumBuffer = elim.makeBlockPairEnumStraight.buffer();

      clCHECK(clSetKernelArg(elimKernel, 0, sizeof(cl_mem), &chainColPtrBuf));
      clCHECK(clSetKernelArg(elimKernel, 1, sizeof(cl_mem), &lumpStartBuf));
      clCHECK(clSetKernelArg(elimKernel, 2, sizeof(cl_mem), &chainRowSpanBuf));
      clCHECK(clSetKernelArg(elimKernel, 3, sizeof(cl_mem), &spanStartBuf));
      clCHECK(clSetKernelArg(elimKernel, 4, sizeof(cl_mem), &chainDataBuf));
      clCHECK(clSetKernelArg(elimKernel, 5, sizeof(cl_mem), &spanToLumpBuf));
      clCHECK(clSetKernelArg(elimKernel, 6, sizeof(cl_mem), &spanOffsetInLumpBuf));
      clCHECK(clSetKernelArg(elimKernel, 7, sizeof(cl_mem), &dataBuffer));
      clCHECK(clSetKernelArg(elimKernel, 8, sizeof(int64_t), &lumpsBegin));
      clCHECK(clSetKernelArg(elimKernel, 9, sizeof(int64_t), &lumpsEnd));
      clCHECK(clSetKernelArg(elimKernel, 10, sizeof(cl_mem), &enumBuffer));
      clCHECK(clSetKernelArg(elimKernel, 11, sizeof(int64_t), &elim.numBlockPairs));

      size_t elimGlobalSize = elim.numBlockPairs;
      clCHECK(clEnqueueNDRangeKernel(queue, elimKernel, 1, nullptr, &elimGlobalSize,
                                     nullptr, 0, nullptr, nullptr));
    }

    OpenCLContext::instance().synchronize();
  }

  virtual void potrf(int64_t n, T* data, int64_t offA) override;
  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override;
  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override;

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    devSpanToChainOffset.load(spanToChainOffset);
  }

  static inline void stridedMatSub(T* dst, int64_t dstStride, const T* src, int64_t srcStride,
                                    int64_t rSize, int64_t cSize) {
    for (int64_t j = 0; j < rSize; j++) {
      for (int64_t i = 0; i < cSize; i++) {
        dst[i] -= src[i];
      }
      dst += dstStride;
      src += srcStride;
    }
  }

  virtual void assemble(T* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + srcColDataOffset;
    const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
    const int64_t* pSpanToChainOffset = spanToChainOffset.data();
    const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();
    const T* matRectPtr = tempBuffer.data();

    // Check if data is on GPU
    auto [buffer, baseOffset] = OpenCLBufferRegistry::instance().findBuffer(data);
    bool onGpu = (buffer != nullptr);

    // For GPU data, we need to download, modify, and upload
    // The writes are scattered, so we download the full data region
    T* workData = data;
    if (onGpu) {
      // Download full data array to cache
      size_t dataSize = skel.dataSize();
      if (cpuDataCache.size() < dataSize) {
        cpuDataCache.resize(dataSize);
      }
      clCHECK(clEnqueueReadBuffer(OpenCLContext::instance().queue(),
                                  buffer, CL_TRUE, baseOffset,
                                  dataSize * sizeof(T), cpuDataCache.data(),
                                  0, nullptr, nullptr));
      workData = cpuDataCache.data();
    }

    // CPU implementation matching reference
    for (int64_t r = 0; r < numBlockRows; r++) {
      int64_t rBegin = chainRowsTillEnd[r - 1] - rectRowBegin;
      int64_t rSize = chainRowsTillEnd[r] - rBegin - rectRowBegin;
      int64_t rParam = pToSpan[r];
      int64_t rOffset = pSpanToChainOffset[rParam];
      const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

      int64_t cEnd = std::min(numBlockCols, r + 1);
      for (int64_t c = 0; c < cEnd; c++) {
        int64_t cStart = chainRowsTillEnd[c - 1] - rectRowBegin;
        int64_t cSize = chainRowsTillEnd[c] - cStart - rectRowBegin;
        int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

        T* dst = workData + offset;
        const T* src = matRowPtr + cStart;
        stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
      }
    }

    // Upload back to GPU if needed
    if (onGpu) {
      size_t dataSize = skel.dataSize();
      clCHECK(clEnqueueWriteBuffer(OpenCLContext::instance().queue(),
                                   buffer, CL_TRUE, baseOffset,
                                   dataSize * sizeof(T), cpuDataCache.data(),
                                   0, nullptr, nullptr));
    }
  }

  OpenCLMirror<T> devTempBuffer;
  OpenCLMirror<int64_t> devSpanToChainOffset;
  vector<int64_t> spanToChainOffset;
  vector<T> tempBuffer;

  const OpenCLSymbolicCtx& sym;
};

// potrf - CPU fallback with GPU sync (CLBlast doesn't have potrf)
template <>
void OpenCLNumericCtx<double>::potrf(int64_t n, double* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<OpenCLSyncOps>(sizeof(double), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  // Download from GPU if needed, perform CPU operation, upload back
  double* cpuData = downloadRegion(data, offA, n * n);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat(cpuData, n, n);
  Eigen::LLT<Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>
      llt(mat);
  uploadRegion(data, offA, n * n);
}

template <>
void OpenCLNumericCtx<float>::potrf(int64_t n, float* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<OpenCLSyncOps>(sizeof(float), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  // Download from GPU if needed, perform CPU operation, upload back
  float* cpuData = downloadRegion(data, offA, n * n);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat(cpuData, n, n);
  Eigen::LLT<Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>
      llt(mat);
  uploadRegion(data, offA, n * n);
}

// trsm - CPU fallback with GPU sync
// TODO: Use CLBlast for GPU trsm when data is on GPU
template <>
void OpenCLNumericCtx<double>::trsm(int64_t n, int64_t k, double* data, int64_t offA,
                                     int64_t offB) {
  auto timer = sym.trsmStat.instance<OpenCLSyncOps>(sizeof(double), n, k);

  // Download both A and B regions at once to avoid overlap issues
  // A is at [offA, offA + n*n), B is at [offB, offB + k*n)
  int64_t regionStart = std::min(offA, offB);
  int64_t regionEnd = std::max(offA + n * n, offB + k * n);
  downloadRegion(data, regionStart, regionEnd - regionStart);

  // Get pointers into the downloaded cache
  double* cpuA = cpuDataCache.data() + offA;
  double* cpuB = cpuDataCache.data() + offB;

  // CPU fallback using Eigen
  // Note: Data is column-major for A (triangular), row-major for B
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      A(cpuA, n, n);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      B(cpuB, k, n);
  A.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(B);

  // Only B is modified
  uploadRegion(data, offB, k * n);
}

template <>
void OpenCLNumericCtx<float>::trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<OpenCLSyncOps>(sizeof(float), n, k);

  // Download both A and B regions at once to avoid overlap issues
  int64_t regionStart = std::min(offA, offB);
  int64_t regionEnd = std::max(offA + n * n, offB + k * n);
  downloadRegion(data, regionStart, regionEnd - regionStart);

  // Get pointers into the downloaded cache
  float* cpuA = cpuDataCache.data() + offA;
  float* cpuB = cpuDataCache.data() + offB;

  // CPU fallback using Eigen
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      A(cpuA, n, n);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      B(cpuB, k, n);
  A.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(B);

  // Only B is modified
  uploadRegion(data, offB, k * n);
}

// saveSyrkGemm - CPU implementation with GPU sync
template <>
void OpenCLNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double* data,
                                             int64_t offset) {
  auto timer = sym.sygeStat.instance<OpenCLSyncOps>(sizeof(double), m, n, k);
  BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

  // Download source data from GPU if needed (read-only)
  // matA is (m, k) and matB is (n, k), both at same offset, so need max(m, n) * k elements
  int64_t rowsNeeded = std::max(m, n);
  const double* AB = downloadRegion(const_cast<double*>(data), offset, rowsNeeded * k);
  double* C = tempBuffer.data();
  Eigen::Map<const MatRMaj<double>> matA(AB, m, k);
  Eigen::Map<const MatRMaj<double>> matB(AB, n, k);
  Eigen::Map<MatRMaj<double>> matC(C, n, m);
  matC.noalias() = matB * matA.transpose();

  sym.gemmCalls++;
}

template <>
void OpenCLNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                                            int64_t offset) {
  auto timer = sym.sygeStat.instance<OpenCLSyncOps>(sizeof(float), m, n, k);
  BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

  // Download source data from GPU if needed (read-only)
  // matA is (m, k) and matB is (n, k), both at same offset, so need max(m, n) * k elements
  int64_t rowsNeeded = std::max(m, n);
  const float* AB = downloadRegion(const_cast<float*>(data), offset, rowsNeeded * k);
  float* C = tempBuffer.data();
  Eigen::Map<const MatRMaj<float>> matA(AB, m, k);
  Eigen::Map<const MatRMaj<float>> matB(AB, n, k);
  Eigen::Map<MatRMaj<float>> matC(C, n, m);
  matC.noalias() = matB * matA.transpose();

  sym.gemmCalls++;
}

template <typename T>
struct OpenCLSolveCtx : SolveCtx<T> {
  OpenCLSolveCtx(const OpenCLSymbolicCtx& sym, int64_t nRHS) : sym(sym), nRHS(nRHS) {
    devSolveBuf.resizeToAtLeast(sym.skel.order() * nRHS);
    // Size tmpBuf for the largest possible sub-block operation
    tmpBuf.resize(sym.skel.order() * nRHS);
  }
  virtual ~OpenCLSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& /* elimData */, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    // CPU fallback implementation
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
      int64_t lumpStart = skel.lumpStart[lump];
      int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
      int64_t colStart = skel.chainColPtr[lump];
      int64_t diagDataPtr = skel.chainData[colStart];

      Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
      OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
      diagBlock.template triangularView<Eigen::Lower>().solveInPlace(matC);

      int64_t colEnd = skel.chainColPtr[lump + 1];
      for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = skel.chainRowSpan[colPtr];
        int64_t rowSpanStart = skel.spanStart[rowSpan];
        int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = skel.chainData[colPtr];
        Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
        OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
        matQ.noalias() -= block * matC;
      }
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /* elimData */, const T* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, T* C, int64_t ldc) override {
    // CPU fallback implementation
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
      int64_t lumpStart = skel.lumpStart[lump];
      int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
      int64_t colStart = skel.chainColPtr[lump];
      int64_t colEnd = skel.chainColPtr[lump + 1];
      OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));

      for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
        int64_t rowSpan = skel.chainRowSpan[colPtr];
        int64_t rowSpanStart = skel.spanStart[rowSpan];
        int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
        int64_t blockPtr = skel.chainData[colPtr];
        Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
        OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
        matC.noalias() -= block.transpose() * matQ;
      }

      int64_t diagDataPtr = skel.chainData[colStart];
      Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
      diagBlock.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
    }
  }

  virtual void symm(const T* data, int64_t offset, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, BaseType<T> alpha) override {
    // CPU fallback: D += alpha * A * C where A is symmetric
    Eigen::Map<const MatRMaj<T>> A(data + offset, n, n);
    OuterStridedCMajMatK<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    OuterStridedCMajMatM<T> matD(D, n, nRHS, OuterStride(ldd));
    matD.noalias() += alpha * A.template selfadjointView<Eigen::Lower>() * matC;
  }

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override {
    Eigen::Map<const MatRMaj<T>> L(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    L.template triangularView<Eigen::Lower>().solveInPlace(matC);
  }

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, BaseType<T> alpha) override {
    // C -= alpha * M * A (where M is at data+offM)
    Eigen::Map<const MatRMaj<T>> M(data + offM, nRows, nCols);
    OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    // Result stored in tmpBuf, applied later via assembleVec
    Eigen::Map<MatRMaj<T>> result(tmpBuf.data(), nRows, nRHS);
    result.noalias() = alpha * M * matA;
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) override {
    const T* A = tmpBuf.data();
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + chainColPtr;
    const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
    int64_t startRow = chainRowsTillEnd[-1];
    for (int64_t i = 0; i < numColItems; i++) {
      int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
      int64_t span = toSpan[i];
      int64_t spanStart = skel.spanStart[span];
      int64_t spanSize = skel.spanStart[span + 1] - spanStart;

      Eigen::Map<const MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize, nRHS);
      OuterStridedCMajMatM<T> matC(C + spanStart, spanSize, nRHS, OuterStride(ldc));
      matC.noalias() += matA;
    }
  }

  virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                       int64_t ldc) override {
    Eigen::Map<const MatRMaj<T>> L(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    L.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
  }

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, BaseType<T> alpha) override {
    // A -= alpha * M^T * C (where C is in tmpBuf from assembleVecT)
    Eigen::Map<const MatRMaj<T>> M(data + offM, nRows, nCols);
    Eigen::Map<const MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
    OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    matA.noalias() -= alpha * M.transpose() * matC;
  }

  virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    T* A = tmpBuf.data();
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + chainColPtr;
    const int64_t* toSpan = skel.chainRowSpan.data() + chainColPtr;
    int64_t startRow = chainRowsTillEnd[-1];
    for (int64_t i = 0; i < numColItems; i++) {
      int64_t rowOffset = chainRowsTillEnd[i - 1] - startRow;
      int64_t span = toSpan[i];
      int64_t spanStart = skel.spanStart[span];
      int64_t spanSize = skel.spanStart[span + 1] - spanStart;

      Eigen::Map<MatRMaj<T>> matA(A + rowOffset * nRHS, spanSize, nRHS);
      OuterStridedCMajMatK<T> matC(C + spanStart, spanSize, nRHS, OuterStride(ldc));
      matA = matC;
    }
  }

  const OpenCLSymbolicCtx& sym;
  int64_t nRHS;
  OpenCLMirror<T> devSolveBuf;
  std::vector<T> tmpBuf;
};

NumericCtxBase* OpenCLSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                            int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new OpenCLNumericCtx<double>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new OpenCLNumericCtx<float>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else {
    // Batched operations not yet supported for OpenCL
    return nullptr;
  }
}

SolveCtxBase* OpenCLSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new OpenCLSolveCtx<double>(*this, nRHS);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new OpenCLSolveCtx<float>(*this, nRHS);
  } else {
    return nullptr;
  }
}

OpsPtr openclOps() { return OpsPtr(new OpenCLOps); }

}  // end namespace BaSpaCho

#endif  // BASPACHO_USE_OPENCL
