/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/Solver.h"
#include <dispenso/parallel_for.h>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include "baspacho/baspacho/ComputationModel.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

Solver::Solver(CoalescedBlockMatrixSkel&& factorSkel_, std::vector<int64_t>&& sparseElimRanges_,
               std::vector<int64_t>&& permutation_, OpsPtr&& ops_, int64_t canFactorUpTo_)
    : factorSkel(std::move(factorSkel_)),
      sparseElimRanges(std::move(sparseElimRanges_)),
      permutation(std::move(permutation_)),
      canFactorUpTo(canFactorUpTo_),
      ops(std::move(ops_)) {
  if (canFactorUpTo < 0) {
    canFactorUpTo = factorSkel.numSpans();
  }
  symCtx = ops->createSymbolicCtx(factorSkel, permutation);
  for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
    elimCtxs.push_back(symCtx->prepareElimination(sparseElimRanges[l], sparseElimRanges[l + 1]));
  }

  initElimination();
}

template <typename T>
void Solver::factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const {
  int64_t lumpStart = factorSkel.lumpStart[lump];
  int64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
  int64_t chainColBegin = factorSkel.chainColPtr[lump];
  int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

  // compute lower diag cholesky dec on diagonal block
  numCtx.potrf(lumpSize, data, diagBlockOffset);

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];
  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
  int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
  int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t numRowsBelowDiag = factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
                             factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  if (numRowsBelowDiag == 0) {
    return;
  }

  numCtx.trsm(lumpSize, numRowsBelowDiag, data, diagBlockOffset, belowDiagOffset);
}

template <typename T>
void Solver::eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const {
  int64_t origLump = factorSkel.boardColLump[ptr];
  int64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

  int64_t origLumpSize = factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
  int64_t chainColBegin = factorSkel.chainColPtr[origLump];

  int64_t boardColBegin = factorSkel.boardColPtr[origLump];
  int64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t belowDiagStart = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  numCtx.saveSyrkGemm(numRowsSub, numRowsFull, origLumpSize, data, belowDiagStart);

  int64_t targetLump = factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
  int64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
  int64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
  int64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
  int64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

  numCtx.assemble(data, rectRowBegin,
                  targetLumpSize,    //
                  srcColDataOffset,  //
                  numRowsSub, numBlockRows, numBlockCols);
}

int64_t Solver::boardElimTempSize(int64_t lump, int64_t boardIndexInCol) const {
  int64_t chainColBegin = factorSkel.chainColPtr[lump];

  int64_t boardColBegin = factorSkel.boardColPtr[lump];
  int64_t boardColEnd = factorSkel.boardColPtr[lump + 1];

  int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
  int64_t rowDataEnd0 = factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
  int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

  int64_t rectRowBegin = factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
  int64_t numRowsSub = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rectRowBegin;
  int64_t numRowsFull = factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rectRowBegin;

  return numRowsSub * numRowsFull;
}

void Solver::initElimination() {
  int64_t denseOpsFromLump = sparseElimRanges.size() ? sparseElimRanges.back() : 0;

  startElimRowPtr.resize(factorSkel.chainColPtr.size() - 1 - denseOpsFromLump);
  maxElimTempSize = 0;
  for (int64_t l = denseOpsFromLump; l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    //  iterate over columns having a non-trivial a-block
    int64_t rPtr0 = factorSkel.boardRowPtr[l];
    int64_t rEnd0 = factorSkel.boardRowPtr[l + 1];
    BASPACHO_CHECK_EQ(factorSkel.boardColLump[rEnd0 - 1], l);
    while (factorSkel.boardColLump[rPtr0] < denseOpsFromLump) {
      rPtr0++;
    }
    BASPACHO_CHECK_LT(rPtr0,
                      rEnd0);  // will stop before end as l > denseOpsFromLump
    startElimRowPtr[l - denseOpsFromLump] = rPtr0;

    for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1];      //
         rPtr < rEnd && factorSkel.boardColLump[rPtr] < l;  //
         rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      int64_t boardIndexInCol = factorSkel.boardColOrd[rPtr];
      int64_t boardSNDataStart = factorSkel.boardColPtr[origLump];
      int64_t boardSNDataEnd = factorSkel.boardColPtr[origLump + 1];
      BASPACHO_CHECK_LT(boardIndexInCol, boardSNDataEnd - boardSNDataStart);
      BASPACHO_CHECK_EQ(l, factorSkel.boardRowLump[boardSNDataStart + boardIndexInCol]);
      maxElimTempSize = max(maxElimTempSize, boardElimTempSize(origLump, boardIndexInCol));
    }
  }
}

template <typename T>
void Solver::factor(T* data, bool verbose) const {
  factorUpTo(data, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::factorUpTo(T* data, int64_t spanIndex, bool verbose) const {
  internalFactorRange(data, 0, spanIndex, verbose);
}

template <typename T>
void Solver::factorFrom(T* data, int64_t spanIndex, bool verbose) const {
  internalFactorRange(data, spanIndex, factorSkel.numSpans(), verbose);
}

template <typename T>
void Solver::internalFactorRange(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                                 bool verbose) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  BASPACHO_CHECK_LE(endSpanIndex, canFactorUpTo);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);

  for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
    if (sparseElimRanges[l + 1] > upToLump) {
      BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
      return;
    } else if (startLump > sparseElimRanges[l]) {
      BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
      continue;
    }
    if (verbose) {
      std::cout << "Elim set: " << l << " (" << sparseElimRanges[l] << ".."
                << sparseElimRanges[l + 1] << ")" << std::endl;
    }
    numCtx->doElimination(*elimCtxs[l], data, sparseElimRanges[l], sparseElimRanges[l + 1]);
  }

  int64_t denseOpsFromLump = sparseElimRanges.empty() ? 0 : sparseElimRanges.back();
  if (verbose) {
    std::cout << "Block-Fact from: " << denseOpsFromLump << std::endl;
  }

  for (int64_t l = std::max(startLump, denseOpsFromLump);
       l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
    numCtx->prepareAssemble(l);

    //  iterate over columns having a non-trivial a-block
    for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1] - 1;  // skip last (diag block)
         rPtr < rEnd; rPtr++) {
      int64_t origLump = factorSkel.boardColLump[rPtr];
      if (origLump >= upToLump) {
        break;
      } else if (origLump < startLump) {
        continue;
      }
      eliminateBoard(*numCtx, data, rPtr);
    }

    if (l < upToLump) {
      factorLump(*numCtx, data, l);
    }
  }
}

template <typename T>
void Solver::solve(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
  internalSolveLtRange(*slvCtx, matData, 0, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  solveLUpTo(matData, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const {
  solveLtUpTo(matData, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLUpTo(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                        int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, 0, spanIndex, vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLtUpTo(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                         int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLtRange(*slvCtx, matData, 0, spanIndex, vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLFrom(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                        int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLRange(*slvCtx, matData, spanIndex, factorSkel.numSpans(), vecData, stride, nRHS);
}

template <typename T>
void Solver::solveLtFrom(const T* matData, int64_t spanIndex, T* vecData, int64_t stride,
                         int nRHS) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
  internalSolveLtRange(*slvCtx, matData, spanIndex, factorSkel.numSpans(), vecData, stride, nRHS);
}

static constexpr bool SparseElimSolve = true;

template <typename T>
void Solver::internalSolveLRange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                 int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump;
  if (SparseElimSolve) {
    for (int64_t l = 0; l + 1 < (int64_t)sparseElimRanges.size(); l++) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_EQ(sparseElimRanges[l], upToLump);
        return;
      } else if (startLump > sparseElimRanges[l]) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        continue;
      }
      slvCtx.sparseElimSolveL(*elimCtxs[l], matData, sparseElimRanges[l], sparseElimRanges[l + 1],
                              vecData, stride);
    }

    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = startLump;
  }

  if (factorSkel.numSpans() == factorSkel.numLumps() && slvCtx.hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx.fragmentedSolveL(matData, denseOpsFromLump, upToLump, vecData);
  } else {
    for (int64_t l = denseOpsFromLump; l < upToLump; l++) {
      int64_t lumpStart = factorSkel.lumpStart[l];
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
      int64_t chainColBegin = factorSkel.chainColPtr[l];
      int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

      slvCtx.solveL(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);

      int64_t boardColBegin = factorSkel.boardColPtr[l];
      int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
      int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
      int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
      int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
      int64_t numRowsBelowDiag =
          factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
          factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
      if (numRowsBelowDiag == 0) {
        continue;
      }

      slvCtx.gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart, stride,
                  -1.0);

      int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
      slvCtx.assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, vecData, stride);
    }
  }
}

template <typename T>
void Solver::internalSolveLtRange(SolveCtx<T>& slvCtx, const T* matData, int64_t startSpanIndex,
                                  int64_t endSpanIndex, T* vecData, int64_t stride,
                                  int nRHS) const {
  BASPACHO_CHECK_GE(startSpanIndex, 0);
  BASPACHO_CHECK_LE(startSpanIndex, endSpanIndex);
  BASPACHO_CHECK_LT(endSpanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[startSpanIndex], 0);
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[endSpanIndex], 0);
  int64_t startLump = factorSkel.spanToLump[startSpanIndex];
  int64_t upToLump = factorSkel.spanToLump[endSpanIndex];

  int64_t denseOpsFromLump;
  if (SparseElimSolve) {
    denseOpsFromLump =
        std::max(startLump, (int64_t)(sparseElimRanges.empty() ? 0 : sparseElimRanges.back()));
  } else {
    denseOpsFromLump = 0;
  }

  int64_t numSpans = factorSkel.lumpToSpan[upToLump] - factorSkel.lumpToSpan[denseOpsFromLump];
  if (numSpans == upToLump - denseOpsFromLump && slvCtx.hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx.fragmentedSolveLt(matData, denseOpsFromLump, upToLump, vecData);
  } else {
    for (int64_t l = upToLump - 1; l >= denseOpsFromLump; l--) {
      int64_t lumpStart = factorSkel.lumpStart[l];
      int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
      int64_t chainColBegin = factorSkel.chainColPtr[l];

      int64_t boardColBegin = factorSkel.boardColPtr[l];
      int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
      int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
      int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
      int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
      int64_t numRowsBelowDiag =
          factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
          factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

      if (numRowsBelowDiag > 0) {
        int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
        slvCtx.assembleVecT(vecData, stride, chainColPtr, numColChains - belowDiagChainColOrd);

        slvCtx.gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, vecData, lumpStart,
                     stride, -1.0);
      }

      int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
      slvCtx.solveLt(matData, diagBlockOffset, lumpSize, vecData, lumpStart, stride);
    }
  }

  if (SparseElimSolve) {
    for (int64_t l = (int64_t)sparseElimRanges.size() - 2; l >= 0; l--) {
      if (sparseElimRanges[l + 1] > upToLump) {
        BASPACHO_CHECK_LE(sparseElimRanges[l], upToLump);
        continue;
      } else if (sparseElimRanges[l] < startLump) {
        BASPACHO_CHECK_GE(startLump, sparseElimRanges[l + 1]);
        return;
      }
      slvCtx.sparseElimSolveLt(*elimCtxs[l], matData, sparseElimRanges[l], sparseElimRanges[l + 1],
                               vecData, stride);
    }
  }
}

template <typename T>
void Solver::addMvFrom(const T* matData, int64_t spanIndex, const T* inVecData, int64_t inStride,
                       T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha) const {
  SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);

  BASPACHO_CHECK_GE(spanIndex, 0);
  BASPACHO_CHECK_LT(spanIndex, (int64_t)factorSkel.spanOffsetInLump.size());
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[spanIndex], 0);
  int64_t startFromLump = factorSkel.spanToLump[spanIndex];
  int64_t denseOpsFromLump = startFromLump;  // sparse ops not supported yet

  int64_t upToLump = factorSkel.lumpStart.size() - 1;

  int64_t numSpans = factorSkel.lumpToSpan[upToLump] - factorSkel.lumpToSpan[denseOpsFromLump];
  if (numSpans == upToLump - denseOpsFromLump && slvCtx->hasFragmentedOps() && nRHS == 1) {
    BASPACHO_CHECK_EQ(factorSkel.lumpToSpan[denseOpsFromLump], denseOpsFromLump);
    slvCtx->fragmentedMV(matData, inVecData, denseOpsFromLump, upToLump, outVecData, alpha);
    return;
  }

  for (int64_t l = denseOpsFromLump; l < upToLump; l++) {
    int64_t lumpStart = factorSkel.lumpStart[l];
    int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[l];

    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
    slvCtx->symm(matData, diagBlockOffset, lumpSize, inVecData, lumpStart, inStride, outVecData,
                 outStride, alpha);

    int64_t boardColBegin = factorSkel.boardColPtr[l];
    int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
    int64_t belowDiagChainColOrd = factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset = factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag =
        factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
      continue;
    }

    int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
    slvCtx->gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, inVecData, lumpStart,
                 inStride, alpha);
    slvCtx->assembleVec(chainColPtr, numColChains - belowDiagChainColOrd, outVecData, outStride);

    slvCtx->assembleVecT(inVecData, inStride, chainColPtr, numColChains - belowDiagChainColOrd);
    slvCtx->gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize, outVecData, lumpStart,
                  outStride, alpha);
  }
}

template <typename T>
void Solver::pseudoFactorFrom(T* data, int64_t spanIndex, bool /* verbose */) const {
  NumericCtxPtr<T> numCtx = symCtx->createNumericCtx<T>(maxElimTempSize, data);
  numCtx->pseudoFactorSpans(data, spanIndex, factorSkel.numSpans());
}

template void Solver::factor<double>(double* data, bool verbose) const;
template void Solver::factor<float>(float* data, bool verbose) const;
template void Solver::factor<vector<double*>>(vector<double*>* data, bool verbose) const;
template void Solver::factor<vector<float*>>(vector<float*>* data, bool verbose) const;
template void Solver::solve<double>(const double* matData, double* vecData, int64_t stride,
                                    int nRHS) const;
template void Solver::solve<float>(const float* matData, float* vecData, int64_t stride,
                                   int nRHS) const;
template void Solver::solve<vector<double*>>(const vector<double*>* matData,
                                             vector<double*>* vecData, int64_t stride,
                                             int nRHS) const;
template void Solver::solve<vector<float*>>(const vector<float*>* matData, vector<float*>* vecData,
                                            int64_t stride, int nRHS) const;
template void Solver::solveL<double>(const double* matData, double* vecData, int64_t stride,
                                     int nRHS) const;
template void Solver::solveL<float>(const float* matData, float* vecData, int64_t stride,
                                    int nRHS) const;
template void Solver::solveL<vector<double*>>(const vector<double*>* matData,
                                              vector<double*>* vecData, int64_t stride,
                                              int nRHS) const;
template void Solver::solveL<vector<float*>>(const vector<float*>* matData, vector<float*>* vecData,
                                             int64_t stride, int nRHS) const;
template void Solver::solveLt<double>(const double* matData, double* vecData, int64_t stride,
                                      int nRHS) const;
template void Solver::solveLt<float>(const float* matData, float* vecData, int64_t stride,
                                     int nRHS) const;
template void Solver::solveLt<vector<double*>>(const vector<double*>* matData,
                                               vector<double*>* vecData, int64_t stride,
                                               int nRHS) const;
template void Solver::solveLt<vector<float*>>(const vector<float*>* matData,
                                              vector<float*>* vecData, int64_t stride,
                                              int nRHS) const;
template void Solver::factorUpTo<double>(double* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorUpTo<float>(float* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorUpTo<vector<double*>>(vector<double*>* data, int64_t spanIndex,
                                                  bool verbose) const;
template void Solver::factorUpTo<vector<float*>>(vector<float*>* data, int64_t spanIndex,
                                                 bool verbose) const;
template void Solver::factorFrom<double>(double* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorFrom<float>(float* data, int64_t spanIndex, bool verbose) const;
template void Solver::factorFrom<vector<double*>>(vector<double*>* data, int64_t spanIndex,
                                                  bool verbose) const;
template void Solver::factorFrom<vector<float*>>(vector<float*>* data, int64_t spanIndex,
                                                 bool verbose) const;
template void Solver::solveLUpTo<double>(const double* matData, int64_t spanIndex, double* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLUpTo<float>(const float* matData, int64_t spanIndex, float* vecData,
                                        int64_t stride, int nRHS) const;
template void Solver::solveLUpTo<vector<double*>>(const vector<double*>* matData, int64_t spanIndex,
                                                  vector<double*>* vecData, int64_t stride,
                                                  int nRHS) const;
template void Solver::solveLUpTo<vector<float*>>(const vector<float*>* matData, int64_t spanIndex,
                                                 vector<float*>* vecData, int64_t stride,
                                                 int nRHS) const;
template void Solver::solveLtUpTo<double>(const double* matData, int64_t spanIndex, double* vecData,
                                          int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<float>(const float* matData, int64_t spanIndex, float* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<vector<double*>>(const vector<double*>* matData,
                                                   int64_t spanIndex, vector<double*>* vecData,
                                                   int64_t stride, int nRHS) const;
template void Solver::solveLtUpTo<vector<float*>>(const vector<float*>* matData, int64_t spanIndex,
                                                  vector<float*>* vecData, int64_t stride,
                                                  int nRHS) const;
template void Solver::addMvFrom<double>(const double* matData, int64_t spanIndex,
                                        const double* inVecData, int64_t inStride,
                                        double* outVecData, int64_t outStride, int nRHS,
                                        double alpha) const;
template void Solver::addMvFrom<float>(const float* matData, int64_t spanIndex,
                                       const float* inVecData, int64_t inStride, float* outVecData,
                                       int64_t outStride, int nRHS, float alpha) const;
template void Solver::pseudoFactorFrom<double>(double* data, int64_t, bool verbose) const;
template void Solver::pseudoFactorFrom<float>(float* data, int64_t, bool verbose) const;
template void Solver::solveLFrom<double>(const double* matData, int64_t spanIndex, double* vecData,
                                         int64_t stride, int nRHS) const;
template void Solver::solveLFrom<float>(const float* matData, int64_t spanIndex, float* vecData,
                                        int64_t stride, int nRHS) const;
template void Solver::solveLtFrom<double>(const double* matData, int64_t spanIndex, double* vecData,
                                          int64_t stride, int nRHS) const;
template void Solver::solveLtFrom<float>(const float* matData, int64_t spanIndex, float* vecData,
                                         int64_t stride, int nRHS) const;

void Solver::printStats() const {
  cout << "Matrix stats:" << endl;
  cout << "  data size......: " << factorSkel.dataSize() << endl;
  cout << "  solve temp data: " << maxElimTempSize << endl;
  if (sparseElimRanges.size() >= 2) {
    cout << "Sparse elimination sets:" << endl;
  }
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    cout << "  elim set [" << sparseElimRanges[l] << ".." << sparseElimRanges[l + 1]
         << "]: " << elimCtxs[l]->elimStat.toString() << endl;
  }
  cout << "Factor timings and call stats:"
       << "\n  largest node size: " << symCtx->potrfBiggestN
       << "\n  potrf: " << symCtx->potrfStat.toString()
       << "\n  trsm: " << symCtx->trsmStat.toString()  //
       << "\n  syrk/gemm(" << symCtx->syrkCalls << "+" << symCtx->gemmCalls
       << "): " << symCtx->sygeStat.toString() << "\n  asmbl: " << symCtx->asmblStat.toString()
       << endl;
  // skip if no solve operation took place:
  if (symCtx->solveSparseLStat.numRuns + symCtx->solveSparseLtStat.numRuns +
          symCtx->solveLStat.numRuns + symCtx->solveLtStat.numRuns >
      0) {
    cout << "Solve timings and call stats:"
         << "\n  solveSparseLStat: " << symCtx->solveSparseLStat.toString()
         << "\n  solveSparseLtStat: " << symCtx->solveSparseLtStat.toString()
         << "\n  solveLStat: " << symCtx->solveLStat.toString()
         << "\n  solveLtStat: " << symCtx->solveLtStat.toString()
         << "\n  solveGemvStat: " << symCtx->solveGemvStat.toString()
         << "\n  solveGemvTStat: " << symCtx->solveGemvTStat.toString()
         << "\n  solveAssVStat: " << symCtx->solveAssVStat.toString()
         << "\n  solveAssVTStat: " << symCtx->solveAssVTStat.toString() << endl;
  }
}

void Solver::enableStats(bool enable) {
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    elimCtxs[l]->elimStat.enabled = enable;
  }
  symCtx->potrfStat.enabled = enable;
  symCtx->trsmStat.enabled = enable;
  symCtx->sygeStat.enabled = enable;
  symCtx->asmblStat.enabled = enable;
}

void Solver::resetStats() {
  for (int64_t l = 0; l < (int64_t)sparseElimRanges.size() - 1; l++) {
    elimCtxs[l]->elimStat.reset();
  }
  symCtx->potrfBiggestN = 0;
  symCtx->potrfStat.reset();
  symCtx->trsmStat.reset();
  symCtx->syrkCalls = 0;
  symCtx->gemmCalls = 0;
  symCtx->sygeStat.reset();
  symCtx->asmblStat.reset();
}

OpsPtr getBackend(const Settings& settings) {
  if (settings.backend == BackendFast) {
    return fastOps(settings.numThreads);
  } else if (settings.backend == BackendCuda) {
#ifdef BASPACHO_USE_CUBLAS
    return cudaOps();
#else
    std::cerr << "Baspacho: CUDA not enabled at compile time" << std::endl;
    abort();
#endif
  }
  BASPACHO_CHECK(settings.backend == BackendRef);
  return simpleOps();
}

SolverPtr createSolver(const Settings& settings, const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss_, const std::vector<int64_t>& sparseElimRanges,
                       const unordered_set<int64_t>& elimLastIds) {
  // no point in providing "elim last" ids if not allowing solve up to such set
  BASPACHO_CHECK(settings.addFillPolicy == AddFillComplete || elimLastIds.empty());

  BASPACHO_CHECK((int64_t)sparseElimRanges.size() != 1);
  int64_t givenSparseElimEnd = sparseElimRanges.empty() ? 0 : sparseElimRanges.back();
  if (!sparseElimRanges.empty()) {
    BASPACHO_CHECK(isStrictlyIncreasing(sparseElimRanges, 0, sparseElimRanges.size()));
    for (int64_t id : elimLastIds) {
      BASPACHO_CHECK_GE(id, givenSparseElimEnd);
    }
  }

  SparseStructure ss = ss_;
  if (settings.addFillPolicy != AddFillNone) {
    for (int64_t e = 0; e < (int64_t)sparseElimRanges.size() - 1; e++) {
      ss = ss.addIndependentEliminationFill(sparseElimRanges[e], sparseElimRanges[e + 1]);
    }
  }

  // create a factor where either no fill is added, either limited for given elims
  if (settings.addFillPolicy == AddFillNone || settings.addFillPolicy == AddFillForGivenElims) {
    vector<int64_t> spanStart;
    spanStart.reserve(paramSize.size() + 1);
    spanStart.insert(spanStart.end(), paramSize.begin(), paramSize.end());
    spanStart.push_back(0);
    cumSumVec(spanStart);

    std::vector<int64_t> lumpToSpan(paramSize.size() + 1);
    ::std::iota(lumpToSpan.begin(), lumpToSpan.end(), 0);

    std::vector<int64_t> permutation(paramSize.size());
    ::std::iota(permutation.begin(), permutation.end(), 0);

    SparseStructure ssT = ss.transpose();  // to csc
    CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, ssT.ptrs, ssT.inds);

    std::vector<int64_t> sparseElimRangesCopy = sparseElimRanges;
    return SolverPtr(new Solver(std::move(factorSkel), std::move(sparseElimRangesCopy),
                                std::move(permutation), getBackend(settings),
                                settings.addFillPolicy == AddFillNone ? 0 : givenSparseElimEnd));
  }

  SparseStructure ssBottom = ss.extractRightBottom(givenSparseElimEnd);

  // find best permutation for right-bottom corner that is left
  vector<int64_t> permutation = ssBottom.fillReducingPermutation();
  vector<int64_t> noCrossPoints;
  if (!elimLastIds.empty()) {  // force those params to go last
    vector<int64_t> parts[2];
    for (int64_t p : permutation) {
      parts[elimLastIds.count(p + givenSparseElimEnd)].push_back(p);
    }
    noCrossPoints.push_back(parts[0].size());
    permutation = parts[0];
    permutation.insert(permutation.end(), parts[1].begin(), parts[1].end());
  }
  vector<int64_t> invPerm = inversePermutation(permutation);
  SparseStructure sortedSsBottom = ssBottom.symmetricPermutation(invPerm, false);

  // apply permutation to param size of right-bottom corner
  std::vector<int64_t> sortedBottomParamSize(paramSize.size() - givenSparseElimEnd);
  for (size_t i = givenSparseElimEnd; i < paramSize.size(); i++) {
    sortedBottomParamSize[invPerm[i - givenSparseElimEnd]] = paramSize[i];
  }

  // auto select computation model (for node merge heuristic), if not provided
  const ComputationModel* compModel = settings.computationModel ? settings.computationModel
                                      : settings.backend == BackendCuda
                                          ? &ComputationModel::model_Cuda117_2080Ti
                                          : &ComputationModel::model_OpenBlas_i7_1185g7;

  // compute as ordinary elimination tree on br-corner
  EliminationTree et(sortedBottomParamSize, sortedSsBottom, compModel);
  et.buildTree();
  et.processTree(settings.findSparseEliminationRanges, noCrossPoints,
                 settings.addFillPolicy == AddFillForAutoElims);
  et.computeAggregateStruct(settings.addFillPolicy == AddFillForAutoElims);

  // ss last rows are to be permuted according to etTotalInvPerm
  vector<int64_t> etTotalInvPerm = composePermutations(et.permInverse, invPerm);
  vector<int64_t> fullInvPerm(givenSparseElimEnd + etTotalInvPerm.size());
  ::std::iota(fullInvPerm.begin(), fullInvPerm.begin() + givenSparseElimEnd, 0);
  for (size_t i = 0; i < etTotalInvPerm.size(); i++) {
    fullInvPerm[i + givenSparseElimEnd] = givenSparseElimEnd + etTotalInvPerm[i];
  }

  // compute span start as cumSum of sorted paramSize
  vector<int64_t> fullSpanStart(paramSize.size() + 1);
  leftPermute(fullSpanStart.begin(), fullInvPerm, paramSize);
  fullSpanStart[paramSize.size()] = 0;
  cumSumVec(fullSpanStart);

  // compute lump to span, knowing up to givenSparseElimEnd it's the identity
  vector<int64_t> fullLumpToSpan;
  fullLumpToSpan.reserve(givenSparseElimEnd + et.lumpToSpan.size());
  fullLumpToSpan.resize(givenSparseElimEnd);
  ::std::iota(fullLumpToSpan.begin(), fullLumpToSpan.begin() + givenSparseElimEnd, 0);
  shiftConcat(fullLumpToSpan, givenSparseElimEnd, et.lumpToSpan.begin(), et.lumpToSpan.end());
  BASPACHO_CHECK_EQ((int64_t)fullSpanStart.size() - 1, fullLumpToSpan.back());

  // matrix with blocks not joined, we will need the first columns
  SparseStructure sortedSsT = ss.symmetricPermutation(fullInvPerm, false).transpose();

  // fullColStart joining sortedSsT.ptrs + shifted elimEndDataPtr
  vector<int64_t> fullColStart;
  fullColStart.reserve(givenSparseElimEnd + et.colStart.size());
  fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                      sortedSsT.ptrs.begin() + givenSparseElimEnd);
  int64_t elimEndDataPtr = sortedSsT.ptrs[givenSparseElimEnd];
  shiftConcat(fullColStart, elimEndDataPtr, et.colStart.begin(), et.colStart.end());
  BASPACHO_CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

  // fullRowParam joining sortedSsT.inds and et.rowParam (moved)
  vector<int64_t> fullRowParam;
  fullRowParam.reserve(elimEndDataPtr + et.rowParam.size());
  fullRowParam.insert(fullRowParam.begin(), sortedSsT.inds.begin(),
                      sortedSsT.inds.begin() + elimEndDataPtr);
  shiftConcat(fullRowParam, givenSparseElimEnd, et.rowParam.begin(), et.rowParam.end());
  BASPACHO_CHECK_EQ((int64_t)fullRowParam.size(), fullColStart.back());

  CoalescedBlockMatrixSkel factorSkel(fullSpanStart, fullLumpToSpan, fullColStart, fullRowParam);

  // include (additional) progressive Schur elimination sets, shifted
  std::vector<int64_t> fullSparseElimRanges = sparseElimRanges;
  if (!et.sparseElimRanges.empty()) {
    shiftConcat(fullSparseElimRanges, givenSparseElimEnd,
                et.sparseElimRanges.begin() + (sparseElimRanges.empty() ? 0 : 1),
                et.sparseElimRanges.end());
  }
  if (fullSparseElimRanges.size() == 1) {
    fullSparseElimRanges.pop_back();
  }
  int64_t fullSparseElimEnd = fullSparseElimRanges.empty() ? 0 : fullSparseElimRanges.back();

  return SolverPtr(new Solver(
      std::move(factorSkel), std::move(fullSparseElimRanges), std::move(fullInvPerm),
      getBackend(settings),
      settings.addFillPolicy == AddFillForAutoElims ? fullSparseElimEnd : paramSize.size()));
}

}  // end namespace BaSpaCho
