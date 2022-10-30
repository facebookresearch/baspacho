/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include <chrono>
#include <iostream>
#include "baspacho/baspacho/CudaAtomic.cuh"
#include "baspacho/baspacho/CudaDefs.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MathUtils.h"
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

// synchronization ops, Cuda version
struct CudaSyncOps {
  static void sync() {
    // can call 'cudaDeviceSynchronize()', but not needed
    cuCHECK(cudaStreamSynchronize(0));
  }
};

struct CudaSymElimCtx : SymElimCtx {
  CudaSymElimCtx() {}
  virtual ~CudaSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  DevMirror<int64_t> makeBlockPairEnumStraight;
};

struct CudaSymbolicCtx : SymbolicCtx {
  CudaSymbolicCtx(const CoalescedBlockMatrixSkel& skel, const std::vector<int64_t>& permutation)
      : skel(skel) {
    // TODO: support custom stream in the future
    cublasCHECK(cublasCreate(&cublasH));
    // cublasCHECK(cublasSetStream(cublasH, stream));
    cusolverCHECK(cusolverDnCreate(&cusolverDnH));
    // cusolverCHECK(cusolverDnSetStream(cusolverDnH, stream));

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

  virtual ~CudaSymbolicCtx() override {
    if (cublasH) {
      cublasCHECK(cublasDestroy(cublasH));
    }
    if (cusolverDnH) {
      cusolverCHECK(cusolverDnDestroy(cusolverDnH));
    }
  }

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr, devSpanToLump.ptr, devLumpStart.ptr, devSpanOffsetInLump.ptr,
              devChainColPtr.ptr, devChainRowSpan.ptr, devChainData.ptr, devPermutation.ptr);
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    CudaSymElimCtx* elim = new CudaSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    // for each lump, compute number of pairs contributing to elim
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
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

  cublasHandle_t cublasH = nullptr;
  cusolverDnHandle_t cusolverDnH = nullptr;

  DevMirror<int64_t> devLumpToSpan;
  DevMirror<int64_t> devChainRowsTillEnd;
  DevMirror<int64_t> devChainRowSpan;
  DevMirror<int64_t> devSpanOffsetInLump;
  DevMirror<int64_t> devLumpStart;
  DevMirror<int64_t> devChainColPtr;
  DevMirror<int64_t> devChainData;
  DevMirror<int64_t> devBoardColPtr;
  DevMirror<int64_t> devBoardChainColOrd;
  DevMirror<int64_t> devSpanStart;
  DevMirror<int64_t> devSpanToLump;
  DevMirror<int64_t> devPermutation;
};

// cuda ops implemented using CUBLAS and custom kernels
struct CudaOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    // cout << "create sym..." << endl;
    return SymbolicCtxPtr(new CudaSymbolicCtx(skel, permutation));
  }
};

template <typename TT, typename B>
__global__ static void factor_lumps_kernel(const int64_t* lumpStart, const int64_t* chainColPtr,
                                           const int64_t* chainData, const int64_t* boardColPtr,
                                           const int64_t* boardChainColOrd,
                                           const int64_t* chainRowsTillEnd, TT* dataB,
                                           int64_t lumpIndexStart, int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }
  int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
  int64_t colStart = chainColPtr[lump];
  int64_t dataPtr = chainData[colStart];

  // in-place lower diag cholesky dec on diagonal block
  T* diagBlockPtr = data + dataPtr;
  cholesky(diagBlockPtr, lumpSize, lumpSize);

  int64_t gatheredStart = boardColPtr[lump];
  int64_t gatheredEnd = boardColPtr[lump + 1];
  int64_t rowDataStart = boardChainColOrd[gatheredStart + 1];
  int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
  int64_t belowDiagStart = chainData[colStart + rowDataStart];
  int64_t numRows =
      chainRowsTillEnd[colStart + rowDataEnd - 1] - chainRowsTillEnd[colStart + rowDataStart - 1];

  T* belowDiagBlockPtr = data + belowDiagStart;
  for (int i = 0; i < numRows; i++) {
    solveUpperT(diagBlockPtr, lumpSize, lumpSize, belowDiagBlockPtr);
    belowDiagBlockPtr += lumpSize;
  }
}

template <typename TT, typename B>
__global__ static void factor_spans_kernel(const int64_t* spanToLump,
                                           const int64_t* spanOffsetInLumpV,
                                           const int64_t* lumpToSpan, const int64_t* spanStart,

                                           const int64_t* lumpStartV, const int64_t* chainColPtr,
                                           const int64_t* chainData, const int64_t* boardColPtr,
                                           const int64_t* boardChainColOrd,
                                           const int64_t* chainRowsTillEnd, TT* dataB,
                                           int64_t spanIndexStart, int64_t spanIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t span = spanIndexStart + i;
  if (span >= spanIndexEnd) {
    return;
  }
  int64_t lump = spanToLump[span];
  int64_t spanOffsetInLump = spanOffsetInLumpV[span];
  int64_t spanIndexInLump = span - lumpToSpan[lump];
  int64_t spanSize = spanStart[span + 1] - spanStart[span];
  int64_t lumpStart = lumpStartV[lump];
  int64_t lumpSize = lumpStartV[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t dataPtr = chainData[colStart + spanIndexInLump] + spanOffsetInLump;

  // in-place lower diag cholesky dec on diagonal block
  T* diagBlockPtr = data + dataPtr;
  cholesky(diagBlockPtr, lumpSize, spanSize);

  int64_t gatheredEnd = boardColPtr[lump + 1];
  int64_t rowDataEnd = boardChainColOrd[gatheredEnd - 1];
  int64_t belowDiagPtr = chainData[colStart + spanIndexInLump + 1] + spanOffsetInLump;
  int64_t numRows =
      chainRowsTillEnd[colStart + rowDataEnd - 1] - chainRowsTillEnd[colStart + spanIndexInLump];

  T* belowDiagBlockPtr = data + belowDiagPtr;
  for (int i = 0; i < numRows; i++) {
    solveUpperT(diagBlockPtr, lumpSize, spanSize, belowDiagBlockPtr);
    belowDiagBlockPtr += lumpSize;
  }
}

template <typename T>
__device__ static inline void do_sparse_elim(const int64_t* chainColPtr, const int64_t* lumpStart,
                                             const int64_t* chainRowSpan, const int64_t* spanStart,
                                             const int64_t* chainData, const int64_t* spanToLump,
                                             const int64_t* spanOffsetInLump, T* data, int64_t l,
                                             int64_t di, int64_t dj) {
  int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
  int64_t lColSize = lumpStart[l + 1] - lumpStart[l];

  int64_t i = startPtr + di;
  int64_t si = chainRowSpan[i];
  int64_t siSize = spanStart[si + 1] - spanStart[si];
  int64_t siDataPtr = chainData[i];
  Eigen::Map<MatRMaj<T>> ilBlock(data + siDataPtr, siSize, lColSize);

  int64_t targetLump = spanToLump[si];
  int64_t targetSpanOffsetInLump = spanOffsetInLump[si];
  int64_t targetStartPtr = chainColPtr[targetLump];  // skip diag block
  int64_t targetEndPtr = chainColPtr[targetLump + 1];
  int64_t targetLumpSize = lumpStart[targetLump + 1] - lumpStart[targetLump];

  int64_t j = startPtr + dj;
  int64_t sj = chainRowSpan[j];
  int64_t sjSize = spanStart[sj + 1] - spanStart[sj];
  int64_t sjDataPtr = chainData[j];

  Eigen::Map<MatRMaj<T>> jlBlock(data + sjDataPtr, sjSize, lColSize);

  uint64_t pos = bisect(chainRowSpan + targetStartPtr, targetEndPtr - targetStartPtr, sj);
  int64_t jiDataPtr = chainData[targetStartPtr + pos];
  OuterStridedMatM<T> jiBlock(data + jiDataPtr + targetSpanOffsetInLump, sjSize, siSize,
                              OuterStride(targetLumpSize));
  // jiBlock -= jlBlock * ilBlock.transpose();
  locked_sub_product(jiBlock, jlBlock, ilBlock);
}

// "naive" elimination kernel, in the sense there is one kernel instance
// per column, and will internally iterate over pairs of blocks (two
// nested loops). Not meant for performance, but as a simpler testing
// version of the below "straigthened" kernel.
template <typename TT, typename B>
__global__ static void sparse_elim_2loops_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart, const int64_t* chainRowSpan,
    const int64_t* spanStart, const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, TT* dataB, int64_t lumpIndexStart, int64_t lumpIndexEnd,
    B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t l = lumpIndexStart + i;
  if (l >= lumpIndexEnd) {
    return;
  }
  int64_t startPtr = chainColPtr[l] + 1;  // skip diag block
  int64_t endPtr = chainColPtr[l + 1];
  for (int64_t i = startPtr; i < endPtr; i++) {
    for (int64_t j = i; j < endPtr; j++) {
      do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart, chainData, spanToLump,
                     spanOffsetInLump, data, l, i - startPtr, j - startPtr);
    }
  }
}

// makeBlockPairEnumStraight contains the cumulated sum of nb*(nb+1)/2
// over all columns, nb being the number of blocks in the columns.
// `i` is the index in the index in the list of all pairs of block in
// the same column. We bisect and get as position the column (relative to
// lumpIndexStart), and as offset to the found value the index in range
// 0..nb*(nb+1)/2-1 in the list of *pairs* of blocks in the column. Such
// index if converted to the ordered pair di/dj
template <typename TT, typename B>
__global__ static void sparse_elim_straight_kernel(
    const int64_t* chainColPtr, const int64_t* lumpStart, const int64_t* chainRowSpan,
    const int64_t* spanStart, const int64_t* chainData, const int64_t* spanToLump,
    const int64_t* spanOffsetInLump, TT* dataB, int64_t lumpIndexStart, int64_t lumpIndexEnd,
    const int64_t* makeBlockPairEnumStraight, int64_t numBlockPairs, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numBlockPairs) {
    return;
  }
  int64_t pos = bisect(makeBlockPairEnumStraight, lumpIndexEnd - lumpIndexStart, i);
  int64_t l = lumpIndexStart + pos;
  int64_t n = chainColPtr[l + 1] - (chainColPtr[l] + 1);
  auto di_dj = toOrderedPair(n, i - makeBlockPairEnumStraight[pos]);
  do_sparse_elim(chainColPtr, lumpStart, chainRowSpan, spanStart, chainData, spanToLump,
                 spanOffsetInLump, data, l, std::get<0>(di_dj), std::get<1>(di_dj));
}

template <typename T>
__device__ static inline void stridedMatSubDev(T* dst, int64_t dstStride, const T* src,
                                               int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    for (uint i = 0; i < cSize; i++) {
      dst[i] -= src[i];
    }
    dst += dstStride;
    src += srcStride;
  }
}

struct Plain {
  __device__ bool verify() { return true; }
  template <typename T>
  __device__ T* get(T* ptr) {
    return ptr;
  }
};

struct Batched {
  int batchSize;
  int batchIndex;
  __device__ bool verify() {
    batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    return batchIndex < batchSize;
  }
  template <typename T>
  __device__ T* get(T* const* ptr) {
    return ptr[batchIndex];
  }
  template <typename T>
  const __device__ T* get(const T* const* ptr) {
    return ptr[batchIndex];
  }
};

template <typename TT, typename B>
__global__ void assemble_kernel(int64_t numBlockRows, int64_t numBlockCols, int64_t rectRowBegin,
                                int64_t srcRectWidth, int64_t dstStride,
                                const int64_t* pChainRowsTillEnd, const int64_t* pToSpan,
                                const int64_t* pSpanToChainOffset, const int64_t* pSpanOffsetInLump,
                                const TT* matRectPtrB, TT* dataB, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* matRectPtr = batch.get(matRectPtrB);
  T* data = batch.get(dataB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numBlockRows * numBlockCols) {
    return;
  }
  int64_t r = i % numBlockRows;
  int64_t c = i / numBlockRows;
  if (c > r) {
    return;
  }

  int64_t rBegin = pChainRowsTillEnd[r - 1] - rectRowBegin;
  int64_t rSize = pChainRowsTillEnd[r] - rBegin - rectRowBegin;
  int64_t rParam = pToSpan[r];
  int64_t rOffset = pSpanToChainOffset[rParam];
  const T* matRowPtr = matRectPtr + rBegin * srcRectWidth;

  int64_t cStart = pChainRowsTillEnd[c - 1] - rectRowBegin;
  int64_t cSize = pChainRowsTillEnd[c] - cStart - rectRowBegin;
  int64_t offset = rOffset + pSpanOffsetInLump[pToSpan[c]];

  T* dst = data + offset;
  const T* src = matRowPtr + cStart;
  stridedMatSubDev(dst, dstStride, src, srcRectWidth, rSize, cSize);
}

template <typename T>
struct CudaNumericCtx : NumericCtx<T> {
  CudaNumericCtx(const CudaSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : spanToChainOffset(numSpans), sym(sym) {
    devTempBuffer.resizeToAtLeast(bufSize);
    devSpanToChainOffset.resizeToAtLeast(spanToChainOffset.size());
  }

  virtual ~CudaNumericCtx() override {}

  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) override {
    auto timer = sym.pseudoFactorStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (spanEnd - spanBegin + wgs - 1) / wgs;
    factor_spans_kernel<T>
        <<<numGroups, wgs>>>(sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr,
                             sym.devLumpToSpan.ptr, sym.devSpanStart.ptr,

                             sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                             sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
                             sym.devChainRowsTillEnd.ptr, data, spanBegin, spanEnd, Plain{});
  }

  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CudaSymElimCtx* pElim = dynamic_cast<const CudaSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CudaSymElimCtx& elim = *pElim;

    auto timer = elim.elimStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    factor_lumps_kernel<T>
        <<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                             sym.devBoardColPtr.ptr, sym.devBoardChainColOrd.ptr,
                             sym.devChainRowsTillEnd.ptr, data, lumpsBegin, lumpsEnd, Plain{});

#if 0
    // double inner loop
    sparse_elim_2loops_kernel<T><<<numGroups, wgs>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
        sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
        sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data,
        lumpsBegin, lumpsEnd, Plain{});
#else
    int wgs2 = 32;
    int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
    sparse_elim_straight_kernel<T><<<numGroups2, wgs2>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr, sym.devChainRowSpan.ptr, sym.devSpanStart.ptr,
        sym.devChainData.ptr, sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, data, lumpsBegin,
        lumpsEnd, elim.makeBlockPairEnumStraight.ptr, elim.numBlockPairs, Plain{});
#endif
  }

  virtual void potrf(int64_t n, T* data, int64_t offA) override;

  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override;

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override;

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // FIXME: compute on CPU and copy, not ideal
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    cuCHECK(cudaMemcpy(devSpanToChainOffset.ptr, spanToChainOffset.data(),
                       spanToChainOffset.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  }

  virtual void assemble(T* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    auto timer = sym.asmblStat.instance<CudaSyncOps>(sizeof(T), numBlockRows, numBlockCols);
    const int64_t* pChainRowsTillEnd = sym.devChainRowsTillEnd.ptr + srcColDataOffset;
    const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
    const int64_t* pSpanToChainOffset = devSpanToChainOffset.ptr;
    const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

    int wgs = 32;
    int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
    assemble_kernel<T><<<numGroups, wgs>>>(
        numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride, pChainRowsTillEnd,
        pToSpan, pSpanToChainOffset, pSpanOffsetInLump, devTempBuffer.ptr, data, Plain{});
  }

  DevMirror<T> devTempBuffer;
  DevMirror<int> devPotrfSingIndex;
  DevMirror<int64_t> devSpanToChainOffset;
  vector<int64_t> spanToChainOffset;

  const CudaSymbolicCtx& sym;
};

template <>
void CudaNumericCtx<double>::potrf(int64_t n, double* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(double), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  int workspaceSize;
  cusolverCHECK(cusolverDnDpotrf_bufferSize(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA,
                                            n, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnDpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA, n,
                                 devTempBuffer.ptr, workspaceSize, devPotrfSingIndex.ptr));

  // TODO: handle/report singularity
  // int info;
  // cuCHECK(cudaMemcpy(&info, devPotrfSingIndex.ptr, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout << info << std::endl;
}

template <>
void CudaNumericCtx<float>::potrf(int64_t n, float* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(float), n);
  sym.potrfBiggestN = max(sym.potrfBiggestN, n);

  int workspaceSize;
  cusolverCHECK(cusolverDnSpotrf_bufferSize(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA,
                                            n, &workspaceSize));

  devTempBuffer.resizeToAtLeast(workspaceSize);
  devPotrfSingIndex.resizeToAtLeast(1);

  cusolverCHECK(cusolverDnSpotrf(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, data + offA, n,
                                 devTempBuffer.ptr, workspaceSize, devPotrfSingIndex.ptr));

  // TODO: handle/report singularity
  // int info;
  // cuCHECK(cudaMemcpy(&info, devPotrfSingIndex.ptr, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout << info << std::endl;
}

template <>
void CudaNumericCtx<double>::trsm(int64_t n, int64_t k, double* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(double), n, k);

  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, k, &alpha, data + offA, n, data + offB, n));
}

template <>
void CudaNumericCtx<float>::trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(float), n, k);

  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, k, &alpha, data + offA, n, data + offB, n));
}

template <>
void CudaNumericCtx<double>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const double* data,
                                          int64_t offset) {
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(double), m, n, k);

  double alpha(1.0), beta(0.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, data + offset, k,
                          data + offset, k, &beta, devTempBuffer.ptr, m));

  sym.gemmCalls++;
}

template <>
void CudaNumericCtx<float>::saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                                         int64_t offset) {
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(float), m, n, k);

  float alpha(1.0), beta(0.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, data + offset, k,
                          data + offset, k, &beta, devTempBuffer.ptr, m));

  sym.gemmCalls++;
}

template <typename T>
string printVec(const vector<T>& ints) {
  stringstream ss;
  ss << "[";
  bool first = true;
  for (auto c : ints) {
    ss << (first ? "" : ", ") << c;
    first = false;
  }
  ss << "]";
  return ss.str();
}

template <typename T>
struct CudaNumericCtx<vector<T*>> : NumericCtx<vector<T*>> {
  CudaNumericCtx(const CudaSymbolicCtx& sym, int64_t bufSize, int64_t numSpans, int batchSize)
      : spanToChainOffset(numSpans), devTempBufs(batchSize, nullptr), sym(sym) {
    devAllJoinedTempBufs.resizeToAtLeast(bufSize * batchSize);
    for (int i = 0; i < batchSize; i++) {
      devTempBufs[i] = devAllJoinedTempBufs.ptr + bufSize * i;
    }
    devTempBufsDev.load(devTempBufs);
    devSpanToChainOffset.resizeToAtLeast(spanToChainOffset.size());
  }

  virtual ~CudaNumericCtx() override {}

  virtual void pseudoFactorSpans(vector<T*>* data, int64_t spanBegin, int64_t spanEnd) override {
    BASPACHO_UNUSED(data, spanBegin, spanEnd);
    throw std::runtime_error("pseudo factor not implemented for batched ops");
  }

  virtual void doElimination(const SymElimCtx& elimData, vector<T*>* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CudaSymElimCtx* pElim = dynamic_cast<const CudaSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CudaSymElimCtx& elim = *pElim;

    auto timer = elim.elimStat.instance<CudaSyncOps>();
    devPtrsA.load(*data, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)data->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (data->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    factor_lumps_kernel<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr, sym.devBoardColPtr.ptr,
        sym.devBoardChainColOrd.ptr, sym.devChainRowsTillEnd.ptr, devPtrsA.ptr, lumpsBegin,
        lumpsEnd, Batched{.batchSize = (int)data->size(), .batchIndex = 0});

#if 0
    // double inner loop
    sparse_elim_2loops_kernel<T*><<<numGroups, wgs>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr,
        sym.devChainRowSpan.ptr, sym.devSpanStart.ptr, sym.devChainData.ptr,
        sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, devPtrsA.ptr,
        lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
#else
    int wgs2 = 32 / batchWgs;
    int numGroups2 = (elim.numBlockPairs + wgs2 - 1) / wgs2;
    dim3 gridDim2(numGroups2, batchGroups);
    dim3 blockDim2(wgs2, batchWgs);
    sparse_elim_straight_kernel<T*><<<gridDim2, blockDim2>>>(
        sym.devChainColPtr.ptr, sym.devLumpStart.ptr, sym.devChainRowSpan.ptr, sym.devSpanStart.ptr,
        sym.devChainData.ptr, sym.devSpanToLump.ptr, sym.devSpanOffsetInLump.ptr, devPtrsA.ptr,
        lumpsBegin, lumpsEnd, elim.makeBlockPairEnumStraight.ptr, elim.numBlockPairs,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
#endif
  }

  virtual void potrf(int64_t n, vector<T*>* data, int64_t offA) override;

  virtual void trsm(int64_t n, int64_t k, vector<T*>* data, int64_t offA, int64_t offB) override;

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const vector<T*>* data,
                            int64_t offset) override;

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // FIXME: compute on CPU and copy, not ideal
    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
    cuCHECK(cudaMemcpy(devSpanToChainOffset.ptr, spanToChainOffset.data(),
                       spanToChainOffset.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  }

  virtual void assemble(vector<T*>* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
    auto timer = sym.asmblStat.instance<CudaSyncOps>(sizeof(T) + data->size() * 100, numBlockRows,
                                                     numBlockCols);
    devPtrsA.load(*data, 0);
    const int64_t* pChainRowsTillEnd = sym.devChainRowsTillEnd.ptr + srcColDataOffset;
    const int64_t* pToSpan = sym.devChainRowSpan.ptr + srcColDataOffset;
    const int64_t* pSpanToChainOffset = devSpanToChainOffset.ptr;
    const int64_t* pSpanOffsetInLump = sym.devSpanOffsetInLump.ptr;

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)data->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (data->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numBlockRows * numBlockCols + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assemble_kernel<T*><<<gridDim, blockDim>>>(
        numBlockRows, numBlockCols, rectRowBegin, srcRectWidth, dstStride, pChainRowsTillEnd,
        pToSpan, pSpanToChainOffset, pSpanOffsetInLump, devTempBufsDev.ptr, devPtrsA.ptr,
        Batched{.batchSize = (int)data->size(), .batchIndex = 0});
  }

  DevMirror<T> devAllJoinedTempBufs;
  vector<T*> devTempBufs;
  DevPtrMirror<T> devTempBufsDev;
  DevMirror<int> devPotrfSingIndex;
  DevPtrMirror<T> devPtrsA, devPtrsB;
  DevMirror<int64_t> devSpanToChainOffset;
  vector<int64_t> spanToChainOffset;

  const CudaSymbolicCtx& sym;
};

template <>
void CudaNumericCtx<vector<double*>>::potrf(int64_t n, vector<double*>* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, n);
  devPtrsA.load(*data, offA);
  devPotrfSingIndex.resizeToAtLeast(data->size());

  cusolverCHECK(cusolverDnDpotrfBatched(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, devPtrsA.ptr, n,
                                        devPotrfSingIndex.ptr, data->size()));

  // TODO: handle/report singularity
  // vector<int> info(data->size());
  // devPotrfSingIndex.get(info);
  // cout << "info: " << printVec(info) << endl;
}

template <>
void CudaNumericCtx<vector<float*>>::potrf(int64_t n, vector<float*>* data, int64_t offA) {
  auto timer = sym.potrfStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, n);
  devPtrsA.load(*data, offA);
  devPotrfSingIndex.resizeToAtLeast(data->size());

  cusolverCHECK(cusolverDnSpotrfBatched(sym.cusolverDnH, CUBLAS_FILL_MODE_UPPER, n, devPtrsA.ptr, n,
                                        devPotrfSingIndex.ptr, data->size()));

  // TODO: handle/report singularity
  // vector<int> info(data->size());
  // devPotrfSingIndex.get(info);
  // cout << "info: " << printVec(info) << endl;
}

template <>
void CudaNumericCtx<vector<double*>>::trsm(int64_t n, int64_t k, vector<double*>* data,
                                           int64_t offA, int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, n, k);
  devPtrsA.load(*data, offA);
  devPtrsB.load(*data, offB);

  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, k, &alpha, devPtrsA.ptr, n, devPtrsB.ptr,
                                 n, data->size()));
}

template <>
void CudaNumericCtx<vector<float*>>::trsm(int64_t n, int64_t k, vector<float*>* data, int64_t offA,
                                          int64_t offB) {
  auto timer = sym.trsmStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, n, k);
  devPtrsA.load(*data, offA);
  devPtrsB.load(*data, offB);

  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, k, &alpha, devPtrsA.ptr, n, devPtrsB.ptr,
                                 n, data->size()));
}

template <>
void CudaNumericCtx<vector<double*>>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                                   const vector<double*>* data, int64_t offset) {
  BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(double) + data->size() * 100, m, n, k);
  devPtrsA.load(*data, offset);
  double alpha(1.0), beta(0.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha,
                                 devPtrsA.ptr, k, devPtrsA.ptr, k, &beta, devTempBufsDev.ptr, m,
                                 data->size()));
  sym.gemmCalls++;
}

template <>
void CudaNumericCtx<vector<float*>>::saveSyrkGemm(int64_t m, int64_t n, int64_t k,
                                                  const vector<float*>* data, int64_t offset) {
  BASPACHO_CHECK_LE(data->size(), devTempBufs.size());
  auto timer = sym.sygeStat.instance<CudaSyncOps>(sizeof(float) + data->size() * 100, m, n, k);
  devPtrsA.load(*data, offset);
  float alpha(1.0), beta(0.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha,
                                 devPtrsA.ptr, k, devPtrsA.ptr, k, &beta, devTempBufsDev.ptr, m,
                                 data->size()));
  sym.gemmCalls++;
}

// SolveCtx helpers
template <typename T>
__device__ static inline void stridedTransAdd(T* dst, int64_t dstStride, const T* src,
                                              int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    T* pDst = dst + j;
    for (uint i = 0; i < cSize; i++) {
      *pDst += src[i];
      pDst += dstStride;
    }
    src += srcStride;
  }
}

template <typename T>
__device__ static inline void stridedTransSet(T* dst, int64_t dstStride, const T* src,
                                              int64_t srcStride, int64_t rSize, int64_t cSize) {
  for (uint j = 0; j < rSize; j++) {
    T* pDst = dst + j;
    for (uint i = 0; i < cSize; i++) {
      *pDst = src[i];
      pDst += dstStride;
    }
    src += srcStride;
  }
}

template <typename TT, typename B>
__global__ void assembleVec_kernel(const int64_t* chainRowsTillEnd, const int64_t* toSpan,
                                   const int64_t* spanStarts, const TT* AB, int64_t numColItems,
                                   TT* CB, int64_t ldc, int64_t nRHS, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(CB)[0])>>;
  T* C = batch.get(CB);
  const T* A = batch.get(AB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numColItems) {
    return;
  }
  int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
  int64_t span = toSpan[i];
  int64_t spanStart = spanStarts[span];
  int64_t spanSize = spanStarts[span + 1] - spanStart;

  stridedTransAdd(C + spanStart, ldc, A + rowOffset * nRHS, nRHS, spanSize, nRHS);
}

template <typename TT, typename B>
__global__ void assembleVecT_kernel(const int64_t* chainRowsTillEnd, const int64_t* toSpan,
                                    const int64_t* spanStarts, const TT* CB, int64_t ldc,
                                    int64_t nRHS, TT* AB, int64_t numColItems, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(AB)[0])>>;
  const T* C = batch.get(CB);
  T* A = batch.get(AB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numColItems) {
    return;
  }
  int64_t rowOffset = chainRowsTillEnd[i - 1] - chainRowsTillEnd[-1];
  int64_t span = toSpan[i];
  int64_t spanStart = spanStarts[span];
  int64_t spanSize = spanStarts[span + 1] - spanStart;

  stridedTransSet(A + rowOffset * nRHS, nRHS, C + spanStart, ldc, nRHS, spanSize);
}

// kernels for sparse-elim solve
template <typename TT, typename B>
__global__ void sparseElim_diagSolveL(const int64_t* lumpStarts, const int64_t* chainColPtr,
                                      const int64_t* chainData, const TT* dataB, TT* vB,
                                      int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                      int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t diagDataPtr = chainData[colStart];

  for (int i = 0; i < nRHS; i++) {
    solveUpperT(data + diagDataPtr, lumpSize, lumpSize, v + lumpStart + ldc * i);
  }
}

template <typename TT, typename B>
__global__ void sparseElim_subDiagMult(const int64_t* lumpStarts, const int64_t* spanStarts,
                                       const int64_t* chainColPtr, const int64_t* chainRowSpan,
                                       const int64_t* chainData, const TT* dataB, TT* vB,
                                       int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                       int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t colEnd = chainColPtr[lump + 1];
  OuterStridedCMajMatM<T> matC(v + lumpStart, lumpSize, nRHS, OuterStride(ldc));

  for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
    int64_t rowSpan = chainRowSpan[colPtr];
    int64_t rowSpanStart = spanStarts[rowSpan];
    int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
    int64_t blockPtr = chainData[colPtr];
    Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
    OuterStridedCMajMatM<T> matQ(v + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
    // matQ -= block * matC;
    locked_sub_AxB(matQ, block, matC);
  }
}

// kernels for sparse-elim solve
template <typename TT, typename B>
__global__ void sparseElim_diagSolveLt(const int64_t* lumpStarts, const int64_t* chainColPtr,
                                       const int64_t* chainData, const TT* dataB, TT* vB,
                                       int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                       int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t diagDataPtr = chainData[colStart];

  for (int i = 0; i < nRHS; i++) {
    solveUpper(data + diagDataPtr, lumpSize, lumpSize, v + lumpStart + ldc * i);
  }
}

template <typename TT, typename B>
__global__ void sparseElim_subDiagMultT(const int64_t* lumpStarts, const int64_t* spanStarts,
                                        const int64_t* chainColPtr, const int64_t* chainRowSpan,
                                        const int64_t* chainData, const TT* dataB, TT* vB,
                                        int64_t ldc, int64_t nRHS, int64_t lumpIndexStart,
                                        int64_t lumpIndexEnd, B batch) {
  if (!batch.verify()) {
    return;
  }
  using T = remove_cv_t<remove_reference_t<decltype(batch.get(dataB)[0])>>;
  const T* data = batch.get(dataB);
  T* v = batch.get(vB);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t lump = lumpIndexStart + i;
  if (lump >= lumpIndexEnd) {
    return;
  }

  int64_t lumpStart = lumpStarts[lump];
  int64_t lumpSize = lumpStarts[lump + 1] - lumpStart;
  int64_t colStart = chainColPtr[lump];
  int64_t colEnd = chainColPtr[lump + 1];
  OuterStridedCMajMatM<T> matC(v + lumpStart, lumpSize, nRHS, OuterStride(ldc));

  for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
    int64_t rowSpan = chainRowSpan[colPtr];
    int64_t rowSpanStart = spanStarts[rowSpan];
    int64_t rowSpanSize = spanStarts[rowSpan + 1] - rowSpanStart;
    int64_t blockPtr = chainData[colPtr];
    Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
    OuterStridedCMajMatM<T> matQ(v + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));
    // matC -= block * matQ;
    locked_sub_ATxB(matC, block, matQ);
  }
}

template <typename T>
struct CudaSolveCtx : SolveCtx<T> {
  CudaSolveCtx(const CudaSymbolicCtx& sym, int64_t nRHS) : sym(sym), nRHS(nRHS) {
    devSolveBuf.resizeToAtLeast(sym.skel.order() * nRHS);
  }
  virtual ~CudaSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    sparseElim_diagSolveL<T><<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr,
                                                 sym.devChainData.ptr, data, C, ldc, nRHS,
                                                 lumpsBegin, lumpsEnd, Plain{});

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMult<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin, lumpsEnd, Plain{});
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance<CudaSyncOps>();

    int wgs = 32;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMultT<T><<<numGroups, wgs>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, data, C, ldc, nRHS, lumpsBegin, lumpsEnd, Plain{});

    sparseElim_diagSolveLt<T><<<numGroups, wgs>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr,
                                                  sym.devChainData.ptr, data, C, ldc, nRHS,
                                                  lumpsBegin, lumpsEnd, Plain{});
  }

  virtual void symm(const T* data, int64_t offset, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, T alpha) override;

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override;

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance<CudaSyncOps>();
    int wgs = 32;
    int numGroups = (numColItems + wgs - 1) / wgs;
    assembleVec_kernel<T><<<numGroups, wgs>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devSolveBuf.ptr, numColItems, C, ldc, nRHS, Plain{});
  }

  virtual void solveLt(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                       int64_t ldc) override;

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, T alpha) override;

  virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance<CudaSyncOps>();
    int wgs = 32;
    int numGroups = (numColItems + wgs - 1) / wgs;
    assembleVecT_kernel<T><<<numGroups, wgs>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, C, ldc, nRHS, devSolveBuf.ptr, numColItems, Plain{});
  }

  const CudaSymbolicCtx& sym;
  int64_t nRHS;
  DevMirror<T> devSolveBuf;
};

template <>
void CudaSolveCtx<double>::symm(const double* data, int64_t offM, int64_t n, const double* C,
                                int64_t offC, int64_t ldc, double* D, int64_t ldd, double alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  double beta(1.0);
  cublasCHECK(cublasDsymm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, nRHS, &alpha,
                          data + offM, n, C + offC, ldc, &beta, D + offC, ldd));
}

template <>
void CudaSolveCtx<float>::symm(const float* data, int64_t offM, int64_t n, const float* C,
                               int64_t offC, int64_t ldc, float* D, int64_t ldd, float alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  float beta(1.0);
  cublasCHECK(cublasSsymm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, nRHS, &alpha,
                          data + offM, n, C + offC, ldc, &beta, D + offC, ldd));
}

template <>
void CudaSolveCtx<double>::solveL(const double* data, int64_t offM, int64_t n, double* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveL(const float* data, int64_t offM, int64_t n, float* C, int64_t offC,
                                 int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemv(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                const double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  double beta(0.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                          A + offA, lda, data + offM, nCols, &beta, devSolveBuf.ptr, nRHS));
}

template <>
void CudaSolveCtx<float>::gemv(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                               const float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  float beta(0.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                          A + offA, lda, data + offM, nCols, &beta, devSolveBuf.ptr, nRHS));
}

template <>
void CudaSolveCtx<double>::solveLt(const double* data, int64_t offM, int64_t n, double* C,
                                   int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  double alpha(1.0);
  cublasCHECK(cublasDtrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<float>::solveLt(const float* data, int64_t offM, int64_t n, float* C,
                                  int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  float alpha(1.0);
  cublasCHECK(cublasStrsm(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, data + offM, n, C + offC, ldc));
}

template <>
void CudaSolveCtx<double>::gemvT(const double* data, int64_t offM, int64_t nRows, int64_t nCols,
                                 double* A, int64_t offA, int64_t lda, double alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  double beta(1.0);
  cublasCHECK(cublasDgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                          data + offM, nCols, devSolveBuf.ptr, nRHS, &beta, A + offA, lda));
}

template <>
void CudaSolveCtx<float>::gemvT(const float* data, int64_t offM, int64_t nRows, int64_t nCols,
                                float* A, int64_t offA, int64_t lda, float alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  float beta(1.0);
  cublasCHECK(cublasSgemm(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                          data + offM, nCols, devSolveBuf.ptr, nRHS, &beta, A + offA, lda));
}

// solve context, batched version
template <typename T>
struct CudaSolveCtx<vector<T*>> : SolveCtx<vector<T*>> {
  CudaSolveCtx(const CudaSymbolicCtx& sym, int64_t nRHS, int batchSize)
      : sym(sym), nRHS(nRHS), devSolveBufs(batchSize, nullptr) {
    int64_t solveBufSize = sym.skel.order() * nRHS;
    devAllJoinedSolveBufs.resizeToAtLeast(batchSize * solveBufSize);
    for (int i = 0; i < batchSize; i++) {
      devSolveBufs[i] = devAllJoinedSolveBufs.ptr + i * solveBufSize;
    }
    devSolveBufsDev.load(devSolveBufs);
  }
  virtual ~CudaSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& /*elimData*/, const vector<T*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, vector<T*>* C,
                                int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance<CudaSyncOps>();

    devPtrsX.load(*data, 0);
    devPtrsY.load(*C, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    sparseElim_diagSolveL<T*>
        <<<gridDim, blockDim>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                                devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
                                Batched{.batchSize = (int)C->size(), .batchIndex = 0});

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMult<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /*elimData*/, const vector<T*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, vector<T*>* C,
                                 int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance<CudaSyncOps>();

    devPtrsX.load(*data, 0);
    devPtrsY.load(*C, 0);

    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (lumpsEnd - lumpsBegin + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);

    // TODO: consider "straightening" inner loop
    sparseElim_subDiagMultT<T*><<<gridDim, blockDim>>>(
        sym.devLumpStart.ptr, sym.devSpanStart.ptr, sym.devChainColPtr.ptr, sym.devChainRowSpan.ptr,
        sym.devChainData.ptr, devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});

    sparseElim_diagSolveLt<T*>
        <<<gridDim, blockDim>>>(sym.devLumpStart.ptr, sym.devChainColPtr.ptr, sym.devChainData.ptr,
                                devPtrsX.ptr, devPtrsY.ptr, ldc, nRHS, lumpsBegin, lumpsEnd,
                                Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void symm(const vector<T*>* data, int64_t offset, int64_t n, const vector<T*>* C,
                    int64_t offC, int64_t ldc, vector<T*>* D, int64_t ldd, T alpha) override;

  virtual void solveL(const vector<T*>* data, int64_t offM, int64_t n, vector<T*>* C, int64_t offC,
                      int64_t ldc) override;

  virtual void gemv(const vector<T*>* data, int64_t offM, int64_t nRows, int64_t nCols,
                    const vector<T*>* A, int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, vector<T*>* C,
                           int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance<CudaSyncOps>();
    devPtrsX.load(*C, 0);
    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numColItems + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assembleVec_kernel<T*><<<gridDim, blockDim>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devSolveBufsDev.ptr, numColItems, devPtrsX.ptr, ldc, nRHS,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  virtual void solveLt(const vector<T*>* data, int64_t offM, int64_t n, vector<T*>* C, int64_t offC,
                       int64_t ldc) override;

  virtual void gemvT(const vector<T*>* data, int64_t offM, int64_t nRows, int64_t nCols,
                     vector<T*>* A, int64_t offA, int64_t lda, T alpha) override;

  virtual void assembleVecT(const vector<T*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance<CudaSyncOps>();
    devPtrsX.load(*C, 0);
    int batchWgs = 32;
    while (batchWgs / 2 >= (int)C->size()) {
      batchWgs /= 2;
    }
    int batchGroups = (C->size() + batchWgs - 1) / batchWgs;
    int wgs = 32 / batchWgs;
    int numGroups = (numColItems + wgs - 1) / wgs;
    dim3 gridDim(numGroups, batchGroups);
    dim3 blockDim(wgs, batchWgs);
    assembleVecT_kernel<T*><<<gridDim, blockDim>>>(
        sym.devChainRowsTillEnd.ptr + chainColPtr, sym.devChainRowSpan.ptr + chainColPtr,
        sym.devSpanStart.ptr, devPtrsX.ptr, ldc, nRHS, devSolveBufsDev.ptr, numColItems,
        Batched{.batchSize = (int)C->size(), .batchIndex = 0});
  }

  const CudaSymbolicCtx& sym;
  int64_t nRHS;
  DevMirror<T> devAllJoinedSolveBufs;
  vector<T*> devSolveBufs;
  DevPtrMirror<T> devSolveBufsDev;
  DevPtrMirror<T> devPtrsX, devPtrsY;
};

template <>
void CudaSolveCtx<vector<double*>>::symm(const vector<double*>* data, int64_t offset, int64_t n,
                                         const vector<double*>* C, int64_t offC, int64_t ldc,
                                         vector<double*>* D, int64_t ldd, double alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  BASPACHO_UNUSED(data, offset, n, C, offC, ldc, D, ldd, alpha);
  throw std::runtime_error("symm not implemented for batched ops");
}

template <>
void CudaSolveCtx<vector<float*>>::symm(const vector<float*>* data, int64_t offset, int64_t n,
                                        const vector<float*>* C, int64_t offC, int64_t ldc,
                                        vector<float*>* D, int64_t ldd, float alpha) {
  auto timer = sym.symmStat.instance<CudaSyncOps>();
  BASPACHO_UNUSED(data, offset, n, C, offC, ldc, D, ldd, alpha);
  throw std::runtime_error("symm not implemented for batched ops");
}

template <>
void CudaSolveCtx<vector<double*>>::solveL(const vector<double*>* data, int64_t offM, int64_t n,
                                           vector<double*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::solveL(const vector<float*>* data, int64_t offM, int64_t n,
                                          vector<float*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::gemv(const vector<double*>* data, int64_t offM, int64_t nRows,
                                         int64_t nCols, const vector<double*>* A, int64_t offA,
                                         int64_t lda, double alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  double beta(0.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                                 devPtrsY.ptr, lda, devPtrsX.ptr, nCols, &beta, devSolveBufsDev.ptr,
                                 nRHS, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::gemv(const vector<float*>* data, int64_t offM, int64_t nRows,
                                        int64_t nCols, const vector<float*>* A, int64_t offA,
                                        int64_t lda, float alpha) {
  auto timer = sym.solveGemvStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  float beta(0.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_C, CUBLAS_OP_N, nRHS, nRows, nCols, &alpha,
                                 devPtrsY.ptr, lda, devPtrsX.ptr, nCols, &beta, devSolveBufsDev.ptr,
                                 nRHS, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::solveLt(const vector<double*>* data, int64_t offM, int64_t n,
                                            vector<double*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  double alpha(1.0);
  cublasCHECK(cublasDtrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::solveLt(const vector<float*>* data, int64_t offM, int64_t n,
                                           vector<float*>* C, int64_t offC, int64_t ldc) {
  auto timer = sym.solveLtStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*C, offC);
  float alpha(1.0);
  cublasCHECK(cublasStrsmBatched(sym.cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, nRHS, &alpha, devPtrsX.ptr, n,
                                 devPtrsY.ptr, ldc, data->size()));
}

template <>
void CudaSolveCtx<vector<double*>>::gemvT(const vector<double*>* data, int64_t offM, int64_t nRows,
                                          int64_t nCols, vector<double*>* A, int64_t offA,
                                          int64_t lda, double alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  double beta(1.0);
  cublasCHECK(cublasDgemmBatched(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                                 devPtrsX.ptr, nCols, devSolveBufsDev.ptr, nRHS, &beta,
                                 devPtrsY.ptr, lda, data->size()));
}

template <>
void CudaSolveCtx<vector<float*>>::gemvT(const vector<float*>* data, int64_t offM, int64_t nRows,
                                         int64_t nCols, vector<float*>* A, int64_t offA,
                                         int64_t lda, float alpha) {
  auto timer = sym.solveGemvTStat.instance<CudaSyncOps>();
  devPtrsX.load(*data, offM);
  devPtrsY.load(*A, offA);
  float beta(1.0);
  cublasCHECK(cublasSgemmBatched(sym.cublasH, CUBLAS_OP_N, CUBLAS_OP_C, nCols, nRHS, nRows, &alpha,
                                 devPtrsX.ptr, nCols, devSolveBufsDev.ptr, nRHS, &beta,
                                 devPtrsY.ptr, lda, data->size()));
}

NumericCtxBase* CudaSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                         int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaNumericCtx<double>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaNumericCtx<float>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == type_index(typeid(vector<double*>))) {
    return new CudaNumericCtx<vector<double*>>(*this, tempBufSize, skel.spanStart.size() - 1,
                                               batchSize);
  } else if (tIdx == type_index(typeid(vector<float*>))) {
    return new CudaNumericCtx<vector<float*>>(*this, tempBufSize, skel.spanStart.size() - 1,
                                              batchSize);
  } else {
    return nullptr;
  }
}

SolveCtxBase* CudaSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  if (tIdx == type_index(typeid(double))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaSolveCtx<double>(*this, nRHS);
  } else if (tIdx == type_index(typeid(float))) {
    BASPACHO_CHECK_EQ(batchSize, 1);
    return new CudaSolveCtx<float>(*this, nRHS);
  } else if (tIdx == type_index(typeid(vector<double*>))) {
    return new CudaSolveCtx<vector<double*>>(*this, nRHS, batchSize);
  } else if (tIdx == type_index(typeid(vector<float*>))) {
    return new CudaSolveCtx<vector<float*>>(*this, nRHS, batchSize);
  } else {
    return nullptr;
  }
}

OpsPtr cudaOps() { return OpsPtr(new CudaOps); }

}  // end namespace BaSpaCho
