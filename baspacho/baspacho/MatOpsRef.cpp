/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <iostream>
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOpsCpuBase.h"
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

struct SimpleSymbolicCtx : CpuBaseSymbolicCtx {
  SimpleSymbolicCtx(const CoalescedBlockMatrixSkel& skel) : CpuBaseSymbolicCtx(skel) {}

  virtual NumericCtxBase* createNumericCtxForType(std::type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(std::type_index tIdx, int nRHS,
                                              int batchSize) override;
};

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& /* permutation */) override {
    return SymbolicCtxPtr(new SimpleSymbolicCtx(skel));
  }
};

template <typename T>
struct SimpleNumericCtx : CpuBaseNumericCtx<T> {
  SimpleNumericCtx(const SimpleSymbolicCtx& sym, int64_t bufSize, int64_t numSpans)
      : CpuBaseNumericCtx<T>(bufSize, numSpans), sym(sym) {}

  virtual void pseudoFactorSpans(T* data, int64_t spanBegin, int64_t spanEnd) override {
    auto timer = sym.pseudoFactorStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t s = spanBegin; s < spanEnd; s++) {
      factorSpan(skel, data, s);
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    const CpuBaseSymElimCtx* pElim = dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CpuBaseSymElimCtx& elim = *pElim;
    auto timer = elim.elimStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      factorLump(skel, data, l);
    }

    int64_t numElimRows = elim.rowPtr.size() - 1;
    int64_t numSpans = skel.spanStart.size() - 1;
    std::vector<int64_t> spanToChainOffset(numSpans);
    for (int64_t sRel = 0UL; sRel < numElimRows; sRel++) {
      eliminateRowChain(elim, skel, data, sRel, spanToChainOffset);
    }
  }

  // This is an alternative of doElimination, which is however written in such
  // a way to mock the sparse elimination as done on the GPU. This is in order
  // to test the logic here, and then create a kernel doing something similar
  void doEliminationMockSparse(const SymElimCtx& elimData, T* data, int64_t lumpsBegin,
                               int64_t lumpsEnd) {
    const CpuBaseSymElimCtx* pElim = dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CpuBaseSymElimCtx& elim = *pElim;
    auto timer = elim.elimStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      factorLump(skel, data, l);
    }

    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t lColSize = skel.lumpStart[l + 1] - skel.lumpStart[l];

      for (int64_t i = startPtr; i < endPtr; i++) {
        int64_t si = skel.chainRowSpan[i];
        int64_t siSize = skel.spanStart[si + 1] - skel.spanStart[si];
        int64_t siDataPtr = skel.chainData[i];
        Eigen::Map<MatRMaj<T>> ilBlock(data + siDataPtr, siSize, lColSize);

        int64_t targetLump = skel.spanToLump[si];
        int64_t targetSpanOffsetInLump = skel.spanOffsetInLump[si];
        int64_t targetStartPtr = skel.chainColPtr[targetLump];  // skip diag block
        int64_t targetEndPtr = skel.chainColPtr[targetLump + 1];
        int64_t targetLumpSize = skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];

        for (int64_t j = i; j < endPtr; j++) {
          int64_t sj = skel.chainRowSpan[j];
          int64_t sjSize = skel.spanStart[sj + 1] - skel.spanStart[sj];
          int64_t sjDataPtr = skel.chainData[j];

          Eigen::Map<MatRMaj<T>> jlBlock(data + sjDataPtr, sjSize, lColSize);

          uint64_t pos =
              bisect(skel.chainRowSpan.data() + targetStartPtr, targetEndPtr - targetStartPtr, sj);
          BASPACHO_CHECK_EQ(skel.chainRowSpan[targetStartPtr + pos], sj);
          int64_t jiDataPtr = skel.chainData[targetStartPtr + pos];
          OuterStridedMatM<T> jiBlock(data + jiDataPtr + targetSpanOffsetInLump, sjSize, siSize,
                                      OuterStride(targetLumpSize));
          jiBlock.noalias() -= jlBlock * ilBlock.transpose();
        }
      }
    }
  }

  virtual void potrf(int64_t n, T* data, int64_t offA) override {
    auto timer = sym.potrfStat.instance(sizeof(T), n);
    sym.potrfBiggestN = std::max(sym.potrfBiggestN, n);

    Eigen::Map<MatRMaj<T>> matA(data + offA, n, n);
    Eigen::LLT<Eigen::Ref<MatRMaj<T>>> llt(matA);
  }

  virtual void trsm(int64_t n, int64_t k, T* data, int64_t offA, int64_t offB) override {
    auto timer = sym.trsmStat.instance(sizeof(T), n, k);

    using MatCMajD = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    // col-major's upper = (row-major's lower).transpose()
    Eigen::Map<const MatCMajD> matA(data + offA, n, n);
    Eigen::Map<MatRMaj<T>> matB(data + offB, k, n);
    matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const T* data,
                            int64_t offset) override {
    auto timer = sym.sygeStat.instance(sizeof(T), m, n, k);
    BASPACHO_CHECK_LE(m * n, (int64_t)tempBuffer.size());

    const T* AB = data + offset;
    T* C = tempBuffer.data();
    Eigen::Map<const MatRMaj<T>> matA(AB, m, k);
    Eigen::Map<const MatRMaj<T>> matB(AB, n, k);
    Eigen::Map<MatRMaj<T>> matC(C, n, m);
    matC.noalias() = matB * matA.transpose();

    sym.gemmCalls++;
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1];
         i < iEnd; i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
  }

  virtual void assemble(T* data, int64_t rectRowBegin,
                        int64_t dstStride,  //
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    auto timer = sym.asmblStat.instance(sizeof(T), numBlockRows, numBlockCols);
    const CoalescedBlockMatrixSkel& skel = sym.skel;
    const int64_t* chainRowsTillEnd = skel.chainRowsTillEnd.data() + srcColDataOffset;
    const int64_t* pToSpan = skel.chainRowSpan.data() + srcColDataOffset;
    const int64_t* pSpanToChainOffset = spanToChainOffset.data();
    const int64_t* pSpanOffsetInLump = skel.spanOffsetInLump.data();

    const T* matRectPtr = tempBuffer.data();

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

        T* dst = data + offset;
        const T* src = matRowPtr + cStart;
        stridedMatSub(dst, dstStride, src, srcRectWidth, rSize, cSize);
      }
    }
  }

  using CpuBaseNumericCtx<T>::factorLump;
  using CpuBaseNumericCtx<T>::factorSpan;
  using CpuBaseNumericCtx<T>::eliminateRowChain;
  using CpuBaseNumericCtx<T>::stridedMatSub;

  using CpuBaseNumericCtx<T>::tempBuffer;
  using CpuBaseNumericCtx<T>::spanToChainOffset;

  const SimpleSymbolicCtx& sym;
};

template <typename T>
struct SimpleSolveCtx : SolveCtx<T> {
  SimpleSolveCtx(const SimpleSymbolicCtx& sym, int nRHS)
      : sym(sym), nRHS(nRHS), tmpBuf(sym.skel.order() * nRHS) {}
  virtual ~SimpleSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const T* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLStat.instance();
    const CpuBaseSymElimCtx* pElim = dynamic_cast<const CpuBaseSymElimCtx*>(&elimData);
    BASPACHO_CHECK_NOTNULL(pElim);
    const CpuBaseSymElimCtx& elim = *pElim;
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t lump = lumpsBegin; lump < lumpsEnd; lump++) {
      int64_t lumpStart = skel.lumpStart[lump];
      int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
      int64_t colStart = skel.chainColPtr[lump];
      int64_t diagDataPtr = skel.chainData[colStart];

      // in-place lower diag cholesky dec on diagonal block
      Eigen::Map<const MatRMaj<T>> diagBlock(data + diagDataPtr, lumpSize, lumpSize);
      OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
      diagBlock.template triangularView<Eigen::Lower>().solveInPlace(matC);

#if 0
            // alternative to the below, keep here for reference
            int64_t colEnd = skel.chainColPtr[lump + 1];
            for (int64_t colPtr = colStart + 1; colPtr < colEnd; colPtr++) {
                int64_t rowSpan = skel.chainRowSpan[colPtr];
                int64_t rowSpanStart = skel.spanStart[rowSpan];
                int64_t rowSpanSize =
                    skel.spanStart[rowSpan + 1] - rowSpanStart;
                int64_t blockPtr = skel.chainData[colPtr];
                Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize,
                                                   lumpSize);
                OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize,
                                             nRHS, OuterStride(ldc));
                matQ.noalias() -= block * matC;
            }
#endif
    }

    // per-row iterations (like during sparse elimination, using elim data):
    // in this way the outer loop can be parallelized
    int64_t numElimRows = elim.rowPtr.size() - 1;
    for (int64_t sRel = 0UL; sRel < numElimRows; sRel++) {
      int64_t rowSpan = sRel + elim.spanRowBegin;
      int64_t rowSpanStart = skel.spanStart[rowSpan];
      int64_t rowSpanSize = skel.spanStart[rowSpan + 1] - rowSpanStart;
      OuterStridedCMajMatM<T> matQ(C + rowSpanStart, rowSpanSize, nRHS, OuterStride(ldc));

      for (int64_t i = elim.rowPtr[sRel], iEnd = elim.rowPtr[sRel + 1]; i < iEnd; i++) {
        int64_t lump = elim.colLump[i];
        int64_t lumpStart = skel.lumpStart[lump];
        int64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
        int64_t chainColOrd = elim.chainColOrd[i];
        BASPACHO_CHECK_GE(chainColOrd,
                          1);  // there must be a diagonal block

        int64_t ptr = skel.chainColPtr[lump] + chainColOrd;
        BASPACHO_CHECK_EQ(skel.chainRowSpan[ptr], rowSpan);
        int64_t blockPtr = skel.chainData[ptr];

        Eigen::Map<const MatRMaj<T>> block(data + blockPtr, rowSpanSize, lumpSize);
        OuterStridedCMajMatM<T> matC(C + lumpStart, lumpSize, nRHS, OuterStride(ldc));
        matQ.noalias() -= block * matC;
      }
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& /*elimData*/, const T* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, T* C, int64_t ldc) override {
    auto timer = sym.solveSparseLtStat.instance();
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    // outer loop can be parallelized
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

  virtual void symm(const T* data, int64_t offM, int64_t n, const T* C, int64_t offC, int64_t ldc,
                    T* D, int64_t ldd, T alpha) override {
    auto timer = sym.symmStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatK<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    OuterStridedCMajMatM<T> matD(D + offC, n, nRHS, OuterStride(ldd));

    matD.noalias() += alpha * (MatRMaj<T>(matA.template selfadjointView<Eigen::Lower>()) * matC);
  }

  virtual void solveL(const T* data, int64_t offM, int64_t n, T* C, int64_t offC,
                      int64_t ldc) override {
    auto timer = sym.solveLStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
  }

  virtual void gemv(const T* data, int64_t offM, int64_t nRows, int64_t nCols, const T* A,
                    int64_t offA, int64_t lda, T alpha) override {
    auto timer = sym.solveGemvStat.instance();
    Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
    OuterStridedCMajMatK<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    Eigen::Map<MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
    matC.noalias() = alpha * (matM * matA);
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, T* C, int64_t ldc) override {
    auto timer = sym.solveAssVStat.instance();
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
    auto timer = sym.solveLtStat.instance();
    Eigen::Map<const MatRMaj<T>> matA(data + offM, n, n);
    OuterStridedCMajMatM<T> matC(C + offC, n, nRHS, OuterStride(ldc));
    matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
  }

  virtual void gemvT(const T* data, int64_t offM, int64_t nRows, int64_t nCols, T* A, int64_t offA,
                     int64_t lda, T alpha) override {
    auto timer = sym.solveGemvTStat.instance();
    Eigen::Map<const MatRMaj<T>> matM(data + offM, nRows, nCols);
    OuterStridedCMajMatM<T> matA(A + offA, nCols, nRHS, OuterStride(lda));
    Eigen::Map<const MatRMaj<T>> matC(tmpBuf.data(), nRows, nRHS);
    matA.noalias() += alpha * (matM.transpose() * matC);
  }

  virtual void assembleVecT(const T* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    auto timer = sym.solveAssVTStat.instance();
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

  const SimpleSymbolicCtx& sym;
  int64_t nRHS;
  vector<T> tmpBuf;
};

NumericCtxBase* SimpleSymbolicCtx::createNumericCtxForType(std::type_index tIdx,
                                                           int64_t tempBufSize, int batchSize) {
  BASPACHO_CHECK_EQ(batchSize, 1);
  if (tIdx == std::type_index(typeid(double))) {
    return new SimpleNumericCtx<double>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else if (tIdx == std::type_index(typeid(float))) {
    return new SimpleNumericCtx<float>(*this, tempBufSize, skel.spanStart.size() - 1);
  } else {
    return nullptr;
  }
}

SolveCtxBase* SimpleSymbolicCtx::createSolveCtxForType(std::type_index tIdx, int nRHS,
                                                       int batchSize) {
  BASPACHO_CHECK_EQ(batchSize, 1);
  if (tIdx == std::type_index(typeid(double))) {
    return new SimpleSolveCtx<double>(*this, nRHS);
  } else if (tIdx == std::type_index(typeid(float))) {
    return new SimpleSolveCtx<float>(*this, nRHS);
  } else {
    return nullptr;
  }
}

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

}  // end namespace BaSpaCho
