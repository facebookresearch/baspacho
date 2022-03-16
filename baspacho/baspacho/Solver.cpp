
#include "Solver.h"

#include <dispenso/parallel_for.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>

#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

Solver::Solver(CoalescedBlockMatrixSkel&& factorSkel_,
               std::vector<int64_t>&& elimLumpRanges_,
               std::vector<int64_t>&& permutation_, OpsPtr&& ops_)
    : factorSkel(std::move(factorSkel_)),
      elimLumpRanges(std::move(elimLumpRanges_)),
      permutation(std::move(permutation_)),
      ops(std::move(ops_)) {
    symCtx = ops->createSymbolicCtx(factorSkel, permutation);
    for (int64_t l = 0; l + 1 < (int64_t)elimLumpRanges.size(); l++) {
        elimCtxs.push_back(symCtx->prepareElimination(elimLumpRanges[l],
                                                      elimLumpRanges[l + 1]));
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
    int64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + 1];
    int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    int64_t belowDiagOffset =
        factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t numRowsBelowDiag =
        factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
        return;
    }

    numCtx.trsm(lumpSize, numRowsBelowDiag, data, diagBlockOffset,
                belowDiagOffset);
}

template <typename T>
void Solver::eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const {
    int64_t origLump = factorSkel.boardColLump[ptr];
    int64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

    int64_t origLumpSize =
        factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
    int64_t chainColBegin = factorSkel.chainColPtr[origLump];

    int64_t boardColBegin = factorSkel.boardColPtr[origLump];
    int64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

    int64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
    int64_t rowDataEnd0 =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
    int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

    int64_t belowDiagStart =
        factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    int64_t rectRowBegin =
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    int64_t numRowsSub =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
        rectRowBegin;
    int64_t numRowsFull =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
        rectRowBegin;

    numCtx.saveSyrkGemm(numRowsSub, numRowsFull, origLumpSize, data,
                        belowDiagStart);

    int64_t targetLump =
        factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
    int64_t targetLumpSize =
        factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
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

    int64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
    int64_t rowDataEnd0 =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
    int64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

    int64_t rectRowBegin =
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    int64_t numRowsSub =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
        rectRowBegin;
    int64_t numRowsFull =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
        rectRowBegin;

    return numRowsSub * numRowsFull;
}

void Solver::initElimination() {
    int64_t denseOpsFromLump =
        elimLumpRanges.size() ? elimLumpRanges[elimLumpRanges.size() - 1] : 0;

    startElimRowPtr.resize(factorSkel.chainColPtr.size() - 1 -
                           denseOpsFromLump);
    maxElimTempSize = 0;
    for (int64_t l = denseOpsFromLump;
         l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
        //  iterate over columns having a non-trivial a-block
        int64_t rPtr = factorSkel.boardRowPtr[l];
        int64_t rEnd = factorSkel.boardRowPtr[l + 1];
        BASPACHO_CHECK_EQ(factorSkel.boardColLump[rEnd - 1], l);
        while (factorSkel.boardColLump[rPtr] < denseOpsFromLump) rPtr++;
        BASPACHO_CHECK_LT(
            rPtr, rEnd);  // will stop before end as l > denseOpsFromLump
        startElimRowPtr[l - denseOpsFromLump] = rPtr;

        for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                     rEnd = factorSkel.boardRowPtr[l + 1];      //
             rPtr < rEnd && factorSkel.boardColLump[rPtr] < l;  //
             rPtr++) {
            int64_t origLump = factorSkel.boardColLump[rPtr];
            int64_t boardIndexInCol = factorSkel.boardColOrd[rPtr];
            int64_t boardSNDataStart = factorSkel.boardColPtr[origLump];
            int64_t boardSNDataEnd = factorSkel.boardColPtr[origLump + 1];
            BASPACHO_CHECK_LT(boardIndexInCol,
                              boardSNDataEnd - boardSNDataStart);
            BASPACHO_CHECK_EQ(
                l, factorSkel.boardRowLump[boardSNDataStart + boardIndexInCol]);
            maxElimTempSize = max(maxElimTempSize,
                                  boardElimTempSize(origLump, boardIndexInCol));
        }
    }
}

template <typename T>
void Solver::factor(T* data, bool verbose) const {
    NumericCtxPtr<T> numCtx =
        symCtx->createNumericCtx<T>(maxElimTempSize, data);

    for (int64_t l = 0; l + 1 < (int64_t)elimLumpRanges.size(); l++) {
        if (verbose) {
            std::cout << "Elim set: " << l << " (" << elimLumpRanges[l] << ".."
                      << elimLumpRanges[l + 1] << ")" << std::endl;
        }
        numCtx->doElimination(*elimCtxs[l], data, elimLumpRanges[l],
                              elimLumpRanges[l + 1]);
    }

    int64_t denseOpsFromLump =
        elimLumpRanges.size() ? elimLumpRanges[elimLumpRanges.size() - 1] : 0;
    if (verbose) {
        std::cout << "Block-Fact from: " << denseOpsFromLump << std::endl;
    }

    for (int64_t l = denseOpsFromLump;
         l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
        numCtx->prepareAssemble(l);

        //  iterate over columns having a non-trivial a-block
        for (int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                     rEnd = factorSkel.boardRowPtr[l + 1] -
                            1;  // skip last (diag block)
             rPtr < rEnd; rPtr++) {
            eliminateBoard(*numCtx, data, rPtr);
        }

        factorLump(*numCtx, data, l);
    }
}

template <typename T>
void Solver::solve(const T* matData, T* vecData, int64_t stride,
                   int nRHS) const {
    SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
    internalSolveL(*slvCtx, matData, vecData, stride);
    internalSolveLt(*slvCtx, matData, vecData, stride);
}

template <typename T>
void Solver::solveL(const T* matData, T* vecData, int64_t stride,
                    int nRHS) const {
    SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
    internalSolveL(*slvCtx, matData, vecData, stride);
}

template <typename T>
void Solver::solveLt(const T* matData, T* vecData, int64_t stride,
                     int nRHS) const {
    SolveCtxPtr<T> slvCtx = symCtx->createSolveCtx<T>(nRHS, matData);
    internalSolveLt(*slvCtx, matData, vecData, stride);
}

static constexpr bool SparseElimSolve = true;

template <typename T>
void Solver::internalSolveL(SolveCtx<T>& slvCtx, const T* matData, T* vecData,
                            int64_t stride) const {
    int64_t denseOpsFromLump;
    if (SparseElimSolve) {
        for (int64_t l = 0; l + 1 < (int64_t)elimLumpRanges.size(); l++) {
            slvCtx.sparseElimSolveL(*elimCtxs[l], matData, elimLumpRanges[l],
                                    elimLumpRanges[l + 1], vecData, stride);
        }

        denseOpsFromLump = elimLumpRanges.size()
                               ? elimLumpRanges[elimLumpRanges.size() - 1]
                               : 0;
    } else {
        denseOpsFromLump = 0;
    }

    for (int64_t l = denseOpsFromLump;
         l < (int64_t)factorSkel.chainColPtr.size() - 1; l++) {
        int64_t lumpStart = factorSkel.lumpStart[l];
        int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
        int64_t chainColBegin = factorSkel.chainColPtr[l];
        int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

        slvCtx.solveL(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                      stride);

        int64_t boardColBegin = factorSkel.boardColPtr[l];
        int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
        int64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + 1];
        int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
        int64_t belowDiagOffset =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        int64_t numRowsBelowDiag =
            factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        if (numRowsBelowDiag == 0) {
            continue;
        }

        slvCtx.gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                    vecData, lumpStart, stride);

        int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
        slvCtx.assembleVec(chainColPtr, numColChains - belowDiagChainColOrd,
                           vecData, stride);
    }
}

template <typename T>
void Solver::internalSolveLt(SolveCtx<T>& slvCtx, const T* matData, T* vecData,
                             int64_t stride) const {
    int64_t denseOpsFromLump;
    if (SparseElimSolve) {
        denseOpsFromLump = elimLumpRanges.size()
                               ? elimLumpRanges[elimLumpRanges.size() - 1]
                               : 0;
    } else {
        denseOpsFromLump = 0;
    }

    for (int64_t l = (int64_t)factorSkel.chainColPtr.size() - 2;
         l >= denseOpsFromLump; l--) {
        int64_t lumpStart = factorSkel.lumpStart[l];
        int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
        int64_t chainColBegin = factorSkel.chainColPtr[l];

        int64_t boardColBegin = factorSkel.boardColPtr[l];
        int64_t boardColEnd = factorSkel.boardColPtr[l + 1];
        int64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + 1];
        int64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
        int64_t belowDiagOffset =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        int64_t numRowsBelowDiag =
            factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

        if (numRowsBelowDiag > 0) {
            int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
            slvCtx.assembleVecT(vecData, stride, chainColPtr,
                                numColChains - belowDiagChainColOrd);

            slvCtx.gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                         vecData, lumpStart, stride);
        }

        int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
        slvCtx.solveLt(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                       stride);
    }

    if (SparseElimSolve) {
        for (int64_t l = (int64_t)elimLumpRanges.size() - 2; l >= 0; l--) {
            slvCtx.sparseElimSolveLt(*elimCtxs[l], matData, elimLumpRanges[l],
                                     elimLumpRanges[l + 1], vecData, stride);
        }
    }
}

template void Solver::factor<double>(double* data, bool verbose) const;
template void Solver::factor<float>(float* data, bool verbose) const;
template void Solver::factor<vector<double*>>(vector<double*>* data,
                                              bool verbose) const;
template void Solver::factor<vector<float*>>(vector<float*>* data,
                                             bool verbose) const;
template void Solver::solve<double>(const double* matData, double* vecData,
                                    int64_t stride, int nRHS) const;
template void Solver::solve<float>(const float* matData, float* vecData,
                                   int64_t stride, int nRHS) const;
template void Solver::solve<vector<double*>>(const vector<double*>* matData,
                                             vector<double*>* vecData,
                                             int64_t stride, int nRHS) const;
template void Solver::solve<vector<float*>>(const vector<float*>* matData,
                                            vector<float*>* vecData,
                                            int64_t stride, int nRHS) const;
template void Solver::solveL<double>(const double* matData, double* vecData,
                                     int64_t stride, int nRHS) const;
template void Solver::solveL<float>(const float* matData, float* vecData,
                                    int64_t stride, int nRHS) const;
template void Solver::solveL<vector<double*>>(const vector<double*>* matData,
                                              vector<double*>* vecData,
                                              int64_t stride, int nRHS) const;
template void Solver::solveL<vector<float*>>(const vector<float*>* matData,
                                             vector<float*>* vecData,
                                             int64_t stride, int nRHS) const;
template void Solver::solveLt<double>(const double* matData, double* vecData,
                                      int64_t stride, int nRHS) const;
template void Solver::solveLt<float>(const float* matData, float* vecData,
                                     int64_t stride, int nRHS) const;
template void Solver::solveLt<vector<double*>>(const vector<double*>* matData,
                                               vector<double*>* vecData,
                                               int64_t stride, int nRHS) const;
template void Solver::solveLt<vector<float*>>(const vector<float*>* matData,
                                              vector<float*>* vecData,
                                              int64_t stride, int nRHS) const;

void Solver::printStats() const {
    cout << "Matrix stats:" << endl;
    cout << "  data size......: " << factorSkel.dataSize() << endl;
    cout << "  solve temp data: " << maxElimTempSize << endl;
    if (elimLumpRanges.size() >= 2) {
        cout << "Sparse elimination sets:" << endl;
    }
    for (int64_t l = 0; l < (int64_t)elimLumpRanges.size() - 1; l++) {
        cout << "  elim set [" << elimLumpRanges[l] << ".."
             << elimLumpRanges[l + 1]
             << "]: " << elimCtxs[l]->elimStat.toString() << endl;
    }
    cout << "Factor timings and call stats:"
         << "\n  largest node size: " << symCtx->potrfBiggestN
         << "\n  potrf: " << symCtx->potrfStat.toString()
         << "\n  trsm: " << symCtx->trsmStat.toString()  //
         << "\n  syrk/gemm(" << symCtx->syrkCalls << "+" << symCtx->gemmCalls
         << "): " << symCtx->sygeStat.toString()
         << "\n  asmbl: " << symCtx->asmblStat.toString() << endl;
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

void Solver::resetStats() {
    for (int64_t l = 0; l < (int64_t)elimLumpRanges.size() - 1; l++) {
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
    if (settings.backend == BackendBlas) {
        return blasOps(settings.numThreads);
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

SolverPtr createSolver(const Settings& settings,
                       const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss) {
    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    std::vector<int64_t> sortedParamSize(paramSize.size());
    leftPermute(sortedParamSize.begin(), invPerm, paramSize);

    EliminationTree et(sortedParamSize, sortedSs);
    et.buildTree();
    et.computeMerges(settings.findSparseEliminationRanges);
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                        et.colStart, et.rowParam);

    vector<int64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    return SolverPtr(new Solver(move(factorSkel), move(et.sparseElimRanges),
                                move(etTotalInvPerm), getBackend(settings)));
}

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<int64_t>& paramSize,
                            const SparseStructure& ss_,
                            const std::vector<int64_t>& elimLumpRanges) {
    BASPACHO_CHECK_GE((int64_t)elimLumpRanges.size(), 2);
    SparseStructure ss =
        ss_.addIndependentEliminationFill(elimLumpRanges[0], elimLumpRanges[1]);
    for (int64_t e = 1; e < (int64_t)elimLumpRanges.size() - 1; e++) {
        ss = ss.addIndependentEliminationFill(elimLumpRanges[e],
                                              elimLumpRanges[e + 1]);
    }

    int64_t elimEnd = elimLumpRanges[elimLumpRanges.size() - 1];
    SparseStructure ssBottom = ss.extractRightBottom(elimEnd);

    // find best permutation for right-bottom corner that is left
    vector<int64_t> permutation = ssBottom.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSsBottom =
        ssBottom.symmetricPermutation(invPerm, false);

    // apply permutation to param size of right-bottom corner
    std::vector<int64_t> sortedBottomParamSize(paramSize.size() - elimEnd);
    for (size_t i = elimEnd; i < paramSize.size(); i++) {
        sortedBottomParamSize[invPerm[i - elimEnd]] = paramSize[i];
    }

    // compute as ordinary elimination tree on br-corner
    EliminationTree et(sortedBottomParamSize, sortedSsBottom);
    et.buildTree();
    et.computeMerges(settings.findSparseEliminationRanges);
    et.computeAggregateStruct();

    // ss last rows are to be permuted according to etTotalInvPerm
    vector<int64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    vector<int64_t> fullInvPerm(elimEnd + etTotalInvPerm.size());
    iota(fullInvPerm.begin(), fullInvPerm.begin() + elimEnd, 0);
    for (size_t i = 0; i < etTotalInvPerm.size(); i++) {
        fullInvPerm[i + elimEnd] = elimEnd + etTotalInvPerm[i];
    }

    // compute span start as cumSum of sorted paramSize
    vector<int64_t> fullSpanStart(paramSize.size() + 1);
    leftPermute(fullSpanStart.begin(), fullInvPerm, paramSize);
    fullSpanStart[paramSize.size()] = 0;
    cumSumVec(fullSpanStart);

    // compute lump to span, knowing up to elimEnd it's the identity
    vector<int64_t> fullLumpToSpan;
    fullLumpToSpan.reserve(elimEnd + et.lumpToSpan.size());
    fullLumpToSpan.resize(elimEnd);
    iota(fullLumpToSpan.begin(), fullLumpToSpan.begin() + elimEnd, 0);
    shiftConcat(fullLumpToSpan, elimEnd, et.lumpToSpan.begin(),
                et.lumpToSpan.end());
    BASPACHO_CHECK_EQ((int64_t)fullSpanStart.size() - 1,
                      fullLumpToSpan[fullLumpToSpan.size() - 1]);

    // matrix with blocks not joined, we will need the first columns
    SparseStructure sortedSsT =
        ss.symmetricPermutation(fullInvPerm, false).transpose();

    // fullColStart joining sortedSsT.ptrs + shifted elimEndDataPtr
    vector<int64_t> fullColStart;
    fullColStart.reserve(elimEnd + et.colStart.size());
    fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                        sortedSsT.ptrs.begin() + elimEnd);
    int64_t elimEndDataPtr = sortedSsT.ptrs[elimEnd];
    shiftConcat(fullColStart, elimEndDataPtr, et.colStart.begin(),
                et.colStart.end());
    BASPACHO_CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

    // fullRowParam joining sortedSsT.inds and et.rowParam (moved)
    vector<int64_t> fullRowParam;
    fullRowParam.reserve(elimEndDataPtr + et.rowParam.size());
    fullRowParam.insert(fullRowParam.begin(), sortedSsT.inds.begin(),
                        sortedSsT.inds.begin() + elimEndDataPtr);
    shiftConcat(fullRowParam, elimEnd, et.rowParam.begin(), et.rowParam.end());
    BASPACHO_CHECK_EQ((int64_t)fullRowParam.size(),
                      fullColStart[fullColStart.size() - 1]);

    CoalescedBlockMatrixSkel factorSkel(fullSpanStart, fullLumpToSpan,
                                        fullColStart, fullRowParam);

    // include (additional) progressive Schur elimination sets, shifted
    std::vector<int64_t> fullElimLumpRanges = elimLumpRanges;
    if (!et.sparseElimRanges.empty()) {
        int64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
        shiftConcat(fullElimLumpRanges, rangeStart, et.sparseElimRanges.begin(),
                    et.sparseElimRanges.end());
    }
    if (fullElimLumpRanges.size() == 1) {
        fullElimLumpRanges.pop_back();
    }

    return SolverPtr(new Solver(move(factorSkel), move(fullElimLumpRanges),
                                move(fullInvPerm), getBackend(settings)));
}

}  // end namespace BaSpaCho