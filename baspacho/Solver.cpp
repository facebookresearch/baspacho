
#include "Solver.h"

#include <dispenso/parallel_for.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>

#include "DebugMacros.h"
#include "EliminationTree.h"
#include "Utils.h"

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
    symCtx = ops->initSymbolicInfo(factorSkel);
    for (int64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
        elimCtxs.push_back(symCtx->prepareElimination(elimLumpRanges[l],
                                                      elimLumpRanges[l + 1]));
    }

    initElimination();
}

void Solver::factorLump(NumericCtx<double>& numCtx, double* data,
                        int64_t lump) const {
    int64_t lumpStart = factorSkel.lumpStart[lump];
    int64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
    int64_t chainColBegin = factorSkel.chainColPtr[lump];
    int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // compute lower diag cholesky dec on diagonal block
    numCtx.potrf(lumpSize, data + diagBlockOffset);

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

    numCtx.trsm(lumpSize, numRowsBelowDiag, data + diagBlockOffset,
                data + belowDiagOffset);
}

void Solver::eliminateBoard(NumericCtx<double>& numCtx, double* data,
                            int64_t ptr) const {
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

void Solver::eliminateBoardBatch(NumericCtx<double>& numCtx, double* data,
                                 int64_t ptr, int64_t batchSize) const {
    int64_t* numRowsSubBatch = (int64_t*)alloca(batchSize * sizeof(int64_t));
    int64_t* numRowsFullBatch = (int64_t*)alloca(batchSize * sizeof(int64_t));
    int64_t* origLumpSizeBatch = (int64_t*)alloca(batchSize * sizeof(int64_t));
    int64_t* belowDiagStartBatch =
        (int64_t*)alloca(batchSize * sizeof(int64_t));
    for (int i = 0; i < batchSize; i++) {
        int64_t origLump = factorSkel.boardColLump[ptr + i];
        int64_t boardIndexInCol = factorSkel.boardColOrd[ptr + i];

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
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        int64_t numRowsSub =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
            rectRowBegin;
        int64_t numRowsFull =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
            rectRowBegin;

        numRowsSubBatch[i] = numRowsSub;
        numRowsFullBatch[i] = numRowsFull;
        origLumpSizeBatch[i] = origLumpSize;
        belowDiagStartBatch[i] = belowDiagStart;
    }

    numCtx.saveSyrkGemmBatched(numRowsSubBatch, numRowsFullBatch,
                               origLumpSizeBatch, data, belowDiagStartBatch,
                               batchSize);

    for (int i = 0; i < batchSize; i++) {
        int64_t origLump = factorSkel.boardColLump[ptr + i];
        int64_t boardIndexInCol = factorSkel.boardColOrd[ptr + i];

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
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        int64_t numRowsSub =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
            rectRowBegin;
        int64_t numRowsFull =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
            rectRowBegin;

        int64_t targetLump =
            factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
        int64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] -
                                 factorSkel.lumpStart[targetLump];
        int64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
        int64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
        int64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

        numCtx.assemble(data, rectRowBegin,
                        targetLumpSize,    //
                        srcColDataOffset,  //
                        numRowsSub, numBlockRows, numBlockCols, i);
    }
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
    for (int64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
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

void Solver::factor(double* data, bool verbose) const {
    if (getenv("BATCHED")) {
        // std::cout << "BATCHED" << std::endl;
        factorXp2(data, verbose);
    } else {
        // std::cout << "SIMPLE" << std::endl;
        factorXp(data, verbose);
    }
}

void Solver::factorXp2(double* data, bool verbose) const {
    int maxBatchSize = max(min(4, 1000000 / (int)(maxElimTempSize + 1)), 1);
    NumericCtxPtr<double> numCtx =
        symCtx->createDoubleContext(maxElimTempSize, maxBatchSize);

    for (int64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
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

    for (int64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
        int64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                rEnd = factorSkel.boardRowPtr[l + 1] -
                       1;  // skip last (diag block)

        numCtx->prepareAssemble(l);

        for (int64_t ptr = rPtr; ptr < rEnd; ptr += maxBatchSize) {
            eliminateBoardBatch(*numCtx, data, ptr,
                                min(maxBatchSize, (int)(rEnd - ptr)));
        }

        factorLump(*numCtx, data, l);
    }
}

void Solver::factorXp(double* data, bool verbose) const {
    int maxBatchSize = max(min(4, 1000000 / (int)(maxElimTempSize + 1)), 1);
    NumericCtxPtr<double> numCtx = symCtx->createDoubleContext(maxElimTempSize);

    for (int64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
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

    for (int64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
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

void Solver::solveL(const double* matData, double* vecData, int64_t stride,
                    int nRHS) const {
    int order = factorSkel.spanStart[factorSkel.spanStart.size() - 1];
    vector<double> tmpData(order * nRHS);

    SolveCtxPtr<double> slvCtx = symCtx->createDoubleSolveContext();
    for (int64_t l = 0; l < factorSkel.chainColPtr.size() - 1; l++) {
        int64_t lumpStart = factorSkel.lumpStart[l];
        int64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
        int64_t chainColBegin = factorSkel.chainColPtr[l];
        int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

        slvCtx->solveL(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                       stride, nRHS);

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

        slvCtx->gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                     vecData, lumpStart, stride, tmpData.data(), nRHS);

        int64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
        slvCtx->assembleVec(tmpData.data(), chainColPtr,
                            numColChains - belowDiagChainColOrd, vecData,
                            stride, nRHS);
    }
}

void Solver::solveLt(const double* matData, double* vecData, int64_t stride,
                     int nRHS) const {
    int order = factorSkel.spanStart[factorSkel.spanStart.size() - 1];
    vector<double> tmpData(order * nRHS);

    SolveCtxPtr<double> slvCtx = symCtx->createDoubleSolveContext();
    for (int64_t l = factorSkel.chainColPtr.size() - 2; (int64_t)l >= 0; l--) {
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
            slvCtx->assembleVecT(vecData, stride, nRHS, tmpData.data(),
                                 chainColPtr,
                                 numColChains - belowDiagChainColOrd);

            slvCtx->gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                          tmpData.data(), nRHS, vecData, lumpStart, stride);
        }

        int64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
        slvCtx->solveLt(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                        stride, nRHS);
    }
}

pair<int64_t, bool> findLargestIndependentLumpSet(
    const CoalescedBlockMatrixSkel& factorSkel, int64_t startLump,
    int64_t maxSize = 8) {
    int64_t limit = kInvalid;
    for (int64_t a = startLump; a < factorSkel.lumpToSpan.size() - 1; a++) {
        if (a >= limit) {
            break;
        }
        if (factorSkel.lumpStart[a + 1] - factorSkel.lumpStart[a] > maxSize) {
            return make_pair(a, true);
        }
        int64_t aPtrStart = factorSkel.boardColPtr[a];
        int64_t aPtrEnd = factorSkel.boardColPtr[a + 1];
        BASPACHO_CHECK_EQ(factorSkel.boardRowLump[aPtrStart], a);
        BASPACHO_CHECK_LE(2, aPtrEnd - aPtrStart);
        limit = min(factorSkel.boardRowLump[aPtrStart + 1], limit);
    }
    return make_pair(min(limit, (int64_t)factorSkel.lumpToSpan.size()), false);
}

SolverPtr createSolver(const Settings& settings,
                       const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss, bool verbose) {
    vector<int64_t> permutation = ss.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    std::vector<int64_t> sortedParamSize(paramSize.size());
    for (size_t i = 0; i < paramSize.size(); i++) {
        sortedParamSize[invPerm[i]] = paramSize[i];
    }

    EliminationTree et(sortedParamSize, sortedSs);
    et.buildTree();
    et.computeMerges();
    et.computeAggregateStruct();

    CoalescedBlockMatrixSkel factorSkel(et.spanStart, et.lumpToSpan,
                                        et.colStart, et.rowParam);

    // find progressive Schur elimination sets
    std::vector<int64_t> elimLumpRanges{0};
    if (settings.findSparseEliminationRanges) {
        while (true) {
            int64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
            auto [rangeEnd, hitSizeLimit] =
                findLargestIndependentLumpSet(factorSkel, rangeStart);
            if (rangeEnd < rangeStart + 10) {
                break;
            }
            if (verbose) {
                std::cout << "Adding indep set: " << rangeStart << ".."
                          << rangeEnd << std::endl;
            }
            elimLumpRanges.push_back(rangeEnd);
            if (hitSizeLimit) {
                break;
            }
        }
    }
    if (elimLumpRanges.size() == 1) {
        elimLumpRanges.pop_back();
    }

    vector<int64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    return SolverPtr(new Solver(move(factorSkel), move(elimLumpRanges),
                                move(etTotalInvPerm),
                                blasOps()  // simpleOps()
                                ));
}

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<int64_t>& paramSize,
                            const SparseStructure& ss_,
                            const std::vector<int64_t>& elimLumpRanges,
                            bool verbose) {
    BASPACHO_CHECK_GE(elimLumpRanges.size(), 2);
    SparseStructure ss =
        ss_.addIndependentEliminationFill(elimLumpRanges[0], elimLumpRanges[1]);
    for (int64_t e = 1; e < elimLumpRanges.size() - 1; e++) {
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
        sortedBottomParamSize[invPerm[i - elimEnd]] = paramSize[i - elimEnd];
    }

    // compute as ordinary elimination tree on br-corner
    EliminationTree et(sortedBottomParamSize, sortedSsBottom);
    et.buildTree();
    et.computeMerges();
    et.computeAggregateStruct();

    // full sorted param size
    std::vector<int64_t> sortedFullParamSize;
    sortedFullParamSize.reserve(paramSize.size());
    sortedFullParamSize.insert(sortedFullParamSize.begin(), paramSize.begin(),
                               paramSize.begin() + elimEnd);
    sortedFullParamSize.insert(sortedFullParamSize.begin(),
                               sortedBottomParamSize.begin(),
                               sortedBottomParamSize.end());

    BASPACHO_CHECK_EQ(et.spanStart.size() - 1,
                      et.lumpToSpan[et.lumpToSpan.size() - 1]);

    // compute span start as cumSum of `sortedFullParamSize`
    vector<int64_t> fullSpanStart;
    fullSpanStart.reserve(sortedFullParamSize.size() + 1);
    fullSpanStart.insert(fullSpanStart.begin(), sortedFullParamSize.begin(),
                         sortedFullParamSize.end());
    fullSpanStart.push_back(0);
    cumSumVec(fullSpanStart);

    // compute lump to span, knowing up to elimEnd it's the identity
    vector<int64_t> fullLumpToSpan(elimEnd + et.lumpToSpan.size());
    iota(fullLumpToSpan.begin(), fullLumpToSpan.begin() + elimEnd, 0);
    for (size_t i = 0; i < et.lumpToSpan.size(); i++) {
        fullLumpToSpan[i + elimEnd] = elimEnd + et.lumpToSpan[i];
    }
    BASPACHO_CHECK_EQ(fullSpanStart.size() - 1,
                      fullLumpToSpan[fullLumpToSpan.size() - 1]);

    // colStart are aggregate lump columns
    // ss last rows are to be permuted according to etTotalInvPerm
    vector<int64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    vector<int64_t> fullInvPerm(elimEnd + etTotalInvPerm.size());
    iota(fullInvPerm.begin(), fullInvPerm.begin() + elimEnd, 0);
    for (size_t i = 0; i < etTotalInvPerm.size(); i++) {
        fullInvPerm[i + elimEnd] = elimEnd + etTotalInvPerm[i];
    }
    SparseStructure sortedSsT =
        ss.symmetricPermutation(fullInvPerm, false).transpose(false);

    // fullColStart joining sortedSsT.ptrs + elimEndDataPtr (shifted)
    vector<int64_t> fullColStart;
    fullColStart.reserve(elimEnd + et.colStart.size());
    fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                        sortedSsT.ptrs.begin() + elimEnd);
    int64_t elimEndDataPtr = sortedSsT.ptrs[elimEnd];
    for (size_t i = 0; i < et.colStart.size(); i++) {
        fullColStart.push_back(elimEndDataPtr + et.colStart[i]);
    }
    BASPACHO_CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

    // fullRowParam joining sortedSsT.inds and et.rowParam (moved)
    vector<int64_t> fullRowParam;
    fullRowParam.reserve(elimEndDataPtr + et.rowParam.size());
    fullRowParam.insert(fullRowParam.begin(), sortedSsT.inds.begin(),
                        sortedSsT.inds.begin() + elimEndDataPtr);
    for (size_t i = 0; i < et.rowParam.size(); i++) {
        fullRowParam.push_back(et.rowParam[i] + elimEnd);
    }
    BASPACHO_CHECK_EQ(fullRowParam.size(),
                      fullColStart[fullColStart.size() - 1]);

    CoalescedBlockMatrixSkel factorSkel(fullSpanStart, fullLumpToSpan,
                                        fullColStart, fullRowParam);

    // find (additional) progressive Schur elimination sets
    std::vector<int64_t> elimLumpRangesArg = elimLumpRanges;
    if (settings.findSparseEliminationRanges) {
        while (true) {
            int64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
            auto [rangeEnd, hitSizeLimit] =
                findLargestIndependentLumpSet(factorSkel, rangeStart);
            if (rangeEnd < rangeStart + 10) {
                break;
            }
            if (verbose) {
                std::cout << "Adding indep set: " << rangeStart << ".."
                          << rangeEnd << std::endl;
            }
            elimLumpRangesArg.push_back(rangeEnd);
            if (hitSizeLimit) {
                break;
            }
        }
    }
    if (elimLumpRangesArg.size() == 1) {
        elimLumpRangesArg.pop_back();
    }

    return SolverPtr(new Solver(move(factorSkel), move(elimLumpRangesArg),
                                move(fullInvPerm),
                                blasOps()  // simpleOps()
                                ));
}