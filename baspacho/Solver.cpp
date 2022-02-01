
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
               std::vector<uint64_t>&& elimLumpRanges_,
               std::vector<uint64_t>&& permutation_, OpsPtr&& ops_)
    : factorSkel(std::move(factorSkel_)),
      elimLumpRanges(std::move(elimLumpRanges_)),
      permutation(std::move(permutation_)),
      ops(std::move(ops_)) {
    symCtx = ops->initSymbolicInfo(factorSkel);
    for (uint64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
        elimCtxs.push_back(symCtx->prepareElimination(elimLumpRanges[l],
                                                      elimLumpRanges[l + 1]));
    }

    initElimination();
}

void Solver::factorLump(NumericCtx<double>& numCtx, double* data,
                        uint64_t lump) const {
    uint64_t lumpStart = factorSkel.lumpStart[lump];
    uint64_t lumpSize = factorSkel.lumpStart[lump + 1] - lumpStart;
    uint64_t chainColBegin = factorSkel.chainColPtr[lump];
    uint64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

    // compute lower diag cholesky dec on diagonal block
    numCtx.potrf(lumpSize, data + diagBlockOffset);

    uint64_t boardColBegin = factorSkel.boardColPtr[lump];
    uint64_t boardColEnd = factorSkel.boardColPtr[lump + 1];
    uint64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + 1];
    uint64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
    uint64_t belowDiagOffset =
        factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t numRowsBelowDiag =
        factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
        return;
    }

    numCtx.trsm(lumpSize, numRowsBelowDiag, data + diagBlockOffset,
                data + belowDiagOffset);
}

void Solver::eliminateBoard(NumericCtx<double>& numCtx, double* data,
                            uint64_t ptr) const {
    uint64_t origLump = factorSkel.boardColLump[ptr];
    uint64_t boardIndexInCol = factorSkel.boardColOrd[ptr];

    uint64_t origLumpSize =
        factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
    uint64_t chainColBegin = factorSkel.chainColPtr[origLump];

    uint64_t boardColBegin = factorSkel.boardColPtr[origLump];
    uint64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

    uint64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
    uint64_t rowDataEnd0 =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
    uint64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

    uint64_t belowDiagStart =
        factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t rectRowBegin =
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    uint64_t numRowsSub =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
        rectRowBegin;
    uint64_t numRowsFull =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
        rectRowBegin;

    numCtx.saveSyrkGemm(numRowsSub, numRowsFull, origLumpSize, data,
                        belowDiagStart);

    uint64_t targetLump =
        factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
    uint64_t targetLumpSize =
        factorSkel.lumpStart[targetLump + 1] - factorSkel.lumpStart[targetLump];
    uint64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
    uint64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
    uint64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

    numCtx.assemble(data, rectRowBegin,
                    targetLumpSize,    //
                    srcColDataOffset,  //
                    numRowsSub, numBlockRows, numBlockCols);
}

void Solver::eliminateBoardBatch(NumericCtx<double>& numCtx, double* data,
                                 uint64_t ptr, uint64_t batchSize) const {
    uint64_t* numRowsSubBatch = (uint64_t*)alloca(batchSize * sizeof(uint64_t));
    uint64_t* numRowsFullBatch =
        (uint64_t*)alloca(batchSize * sizeof(uint64_t));
    uint64_t* origLumpSizeBatch =
        (uint64_t*)alloca(batchSize * sizeof(uint64_t));
    uint64_t* belowDiagStartBatch =
        (uint64_t*)alloca(batchSize * sizeof(uint64_t));
    for (int i = 0; i < batchSize; i++) {
        uint64_t origLump = factorSkel.boardColLump[ptr + i];
        uint64_t boardIndexInCol = factorSkel.boardColOrd[ptr + i];

        uint64_t origLumpSize =
            factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
        uint64_t chainColBegin = factorSkel.chainColPtr[origLump];

        uint64_t boardColBegin = factorSkel.boardColPtr[origLump];
        uint64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

        uint64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
        uint64_t rowDataEnd0 =
            factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
        uint64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

        uint64_t belowDiagStart =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        uint64_t rectRowBegin =
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        uint64_t numRowsSub =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
            rectRowBegin;
        uint64_t numRowsFull =
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
        uint64_t origLump = factorSkel.boardColLump[ptr + i];
        uint64_t boardIndexInCol = factorSkel.boardColOrd[ptr + i];

        uint64_t origLumpSize =
            factorSkel.lumpStart[origLump + 1] - factorSkel.lumpStart[origLump];
        uint64_t chainColBegin = factorSkel.chainColPtr[origLump];

        uint64_t boardColBegin = factorSkel.boardColPtr[origLump];
        uint64_t boardColEnd = factorSkel.boardColPtr[origLump + 1];

        uint64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
        uint64_t rowDataEnd0 =
            factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
        uint64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

        uint64_t belowDiagStart =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        uint64_t rectRowBegin =
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        uint64_t numRowsSub =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
            rectRowBegin;
        uint64_t numRowsFull =
            factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
            rectRowBegin;

        uint64_t targetLump =
            factorSkel.boardRowLump[boardColBegin + boardIndexInCol];
        uint64_t targetLumpSize = factorSkel.lumpStart[targetLump + 1] -
                                  factorSkel.lumpStart[targetLump];
        uint64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
        uint64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
        uint64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;

        numCtx.assemble(data, rectRowBegin,
                        targetLumpSize,    //
                        srcColDataOffset,  //
                        numRowsSub, numBlockRows, numBlockCols, i);
    }
}

uint64_t Solver::boardElimTempSize(uint64_t lump,
                                   uint64_t boardIndexInCol) const {
    uint64_t chainColBegin = factorSkel.chainColPtr[lump];

    uint64_t boardColBegin = factorSkel.boardColPtr[lump];
    uint64_t boardColEnd = factorSkel.boardColPtr[lump + 1];

    uint64_t belowDiagChainColOrd =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol];
    uint64_t rowDataEnd0 =
        factorSkel.boardChainColOrd[boardColBegin + boardIndexInCol + 1];
    uint64_t rowDataEnd1 = factorSkel.boardChainColOrd[boardColEnd - 1];

    uint64_t rectRowBegin =
        factorSkel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    uint64_t numRowsSub =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] -
        rectRowBegin;
    uint64_t numRowsFull =
        factorSkel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] -
        rectRowBegin;

    return numRowsSub * numRowsFull;
}

void Solver::initElimination() {
    uint64_t denseOpsFromLump =
        elimLumpRanges.size() ? elimLumpRanges[elimLumpRanges.size() - 1] : 0;

    startElimRowPtr.resize(factorSkel.chainColPtr.size() - 1 -
                           denseOpsFromLump);
    maxElimTempSize = 0;
    for (uint64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
        //  iterate over columns having a non-trivial a-block
        uint64_t rPtr = factorSkel.boardRowPtr[l];
        uint64_t rEnd = factorSkel.boardRowPtr[l + 1];
        BASPACHO_CHECK_EQ(factorSkel.boardColLump[rEnd - 1], l);
        while (factorSkel.boardColLump[rPtr] < denseOpsFromLump) rPtr++;
        BASPACHO_CHECK_LT(
            rPtr, rEnd);  // will stop before end as l > denseOpsFromLump
        startElimRowPtr[l - denseOpsFromLump] = rPtr;

        for (uint64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                      rEnd = factorSkel.boardRowPtr[l + 1];     //
             rPtr < rEnd && factorSkel.boardColLump[rPtr] < l;  //
             rPtr++) {
            uint64_t origLump = factorSkel.boardColLump[rPtr];
            uint64_t boardIndexInCol = factorSkel.boardColOrd[rPtr];
            uint64_t boardSNDataStart = factorSkel.boardColPtr[origLump];
            uint64_t boardSNDataEnd = factorSkel.boardColPtr[origLump + 1];
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

    for (uint64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
        if (verbose) {
            std::cout << "Elim set: " << l << " (" << elimLumpRanges[l] << ".."
                      << elimLumpRanges[l + 1] << ")" << std::endl;
        }
        numCtx->doElimination(*elimCtxs[l], data, elimLumpRanges[l],
                              elimLumpRanges[l + 1]);
    }

    uint64_t denseOpsFromLump =
        elimLumpRanges.size() ? elimLumpRanges[elimLumpRanges.size() - 1] : 0;
    if (verbose) {
        std::cout << "Block-Fact from: " << denseOpsFromLump << std::endl;
    }

    for (uint64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
        uint64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                 rEnd = factorSkel.boardRowPtr[l + 1] -
                        1;  // skip last (diag block)

        numCtx->prepareAssemble(l);

        for (uint64_t ptr = rPtr; ptr < rEnd; ptr += maxBatchSize) {
            eliminateBoardBatch(*numCtx, data, ptr,
                                min(maxBatchSize, (int)(rEnd - ptr)));
        }

        factorLump(*numCtx, data, l);
    }
}

void Solver::factorXp(double* data, bool verbose) const {
    int maxBatchSize = max(min(4, 1000000 / (int)(maxElimTempSize + 1)), 1);
    NumericCtxPtr<double> numCtx =
        symCtx->createDoubleContext(maxElimTempSize, maxBatchSize);

    for (uint64_t l = 0; l + 1 < elimLumpRanges.size(); l++) {
        if (verbose) {
            std::cout << "Elim set: " << l << " (" << elimLumpRanges[l] << ".."
                      << elimLumpRanges[l + 1] << ")" << std::endl;
        }
        numCtx->doElimination(*elimCtxs[l], data, elimLumpRanges[l],
                              elimLumpRanges[l + 1]);
    }

    uint64_t denseOpsFromLump =
        elimLumpRanges.size() ? elimLumpRanges[elimLumpRanges.size() - 1] : 0;
    if (verbose) {
        std::cout << "Block-Fact from: " << denseOpsFromLump << std::endl;
    }

    for (uint64_t l = denseOpsFromLump; l < factorSkel.chainColPtr.size() - 1;
         l++) {
        numCtx->prepareAssemble(l);

        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = startElimRowPtr[l - denseOpsFromLump],
                      rEnd = factorSkel.boardRowPtr[l + 1] -
                             1;  // skip last (diag block)
             rPtr < rEnd; rPtr++) {
            eliminateBoard(*numCtx, data, rPtr);
        }

        factorLump(*numCtx, data, l);
    }
}

void Solver::solveL(const double* matData, double* vecData, int stride,
                    int nRHS) const {
    int order = factorSkel.spanStart[factorSkel.spanStart.size() - 1];
    vector<double> tmpData(order * nRHS);

    SolveCtxPtr<double> slvCtx = symCtx->createDoubleSolveContext();
    for (uint64_t l = 0; l < factorSkel.chainColPtr.size() - 1; l++) {
        uint64_t lumpStart = factorSkel.lumpStart[l];
        uint64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
        uint64_t chainColBegin = factorSkel.chainColPtr[l];
        uint64_t diagBlockOffset = factorSkel.chainData[chainColBegin];

        slvCtx->solveL(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                       stride, nRHS);

        uint64_t boardColBegin = factorSkel.boardColPtr[l];
        uint64_t boardColEnd = factorSkel.boardColPtr[l + 1];
        uint64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + 1];
        uint64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
        uint64_t belowDiagOffset =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        uint64_t numRowsBelowDiag =
            factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
        if (numRowsBelowDiag == 0) {
            continue;
        }

        slvCtx->gemv(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                     vecData, lumpStart, stride, tmpData.data(), nRHS);

        uint64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
        slvCtx->assembleVec(tmpData.data(), chainColPtr,
                            numColChains - belowDiagChainColOrd, vecData,
                            stride, nRHS);
    }
}

void Solver::solveLt(const double* matData, double* vecData, int stride,
                     int nRHS) const {
    int order = factorSkel.spanStart[factorSkel.spanStart.size() - 1];
    vector<double> tmpData(order * nRHS);

    SolveCtxPtr<double> slvCtx = symCtx->createDoubleSolveContext();
    for (uint64_t l = factorSkel.chainColPtr.size() - 2; (int64_t)l >= 0; l--) {
        uint64_t lumpStart = factorSkel.lumpStart[l];
        uint64_t lumpSize = factorSkel.lumpStart[l + 1] - lumpStart;
        uint64_t chainColBegin = factorSkel.chainColPtr[l];

        uint64_t boardColBegin = factorSkel.boardColPtr[l];
        uint64_t boardColEnd = factorSkel.boardColPtr[l + 1];
        uint64_t belowDiagChainColOrd =
            factorSkel.boardChainColOrd[boardColBegin + 1];
        uint64_t numColChains = factorSkel.boardChainColOrd[boardColEnd - 1];
        uint64_t belowDiagOffset =
            factorSkel.chainData[chainColBegin + belowDiagChainColOrd];
        uint64_t numRowsBelowDiag =
            factorSkel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
            factorSkel
                .chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

        if (numRowsBelowDiag > 0) {
            uint64_t chainColPtr = chainColBegin + belowDiagChainColOrd;
            slvCtx->assembleVecT(vecData, stride, nRHS, tmpData.data(),
                                 chainColPtr,
                                 numColChains - belowDiagChainColOrd);

            slvCtx->gemvT(matData, belowDiagOffset, numRowsBelowDiag, lumpSize,
                          tmpData.data(), nRHS, vecData, lumpStart, stride);
        }

        uint64_t diagBlockOffset = factorSkel.chainData[chainColBegin];
        slvCtx->solveLt(matData, diagBlockOffset, lumpSize, vecData, lumpStart,
                        stride, nRHS);
    }
}

pair<uint64_t, bool> findLargestIndependentLumpSet(
    const CoalescedBlockMatrixSkel& factorSkel, uint64_t startLump,
    uint64_t maxSize = 8) {
    uint64_t limit = kInvalid;
    for (uint64_t a = startLump; a < factorSkel.lumpToSpan.size() - 1; a++) {
        if (a >= limit) {
            break;
        }
        if (factorSkel.lumpStart[a + 1] - factorSkel.lumpStart[a] > maxSize) {
            return make_pair(a, true);
        }
        uint64_t aPtrStart = factorSkel.boardColPtr[a];
        uint64_t aPtrEnd = factorSkel.boardColPtr[a + 1];
        BASPACHO_CHECK_EQ(factorSkel.boardRowLump[aPtrStart], a);
        BASPACHO_CHECK_LE(2, aPtrEnd - aPtrStart);
        limit = min(factorSkel.boardRowLump[aPtrStart + 1], limit);
    }
    return make_pair(min(limit, factorSkel.lumpToSpan.size()), false);
}

SolverPtr createSolver(const Settings& settings,
                       const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss, bool verbose) {
    vector<uint64_t> permutation = ss.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    std::vector<uint64_t> sortedParamSize(paramSize.size());
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
    std::vector<uint64_t> elimLumpRanges{0};
    if (settings.findSparseEliminationRanges) {
        while (true) {
            uint64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
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

    vector<uint64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    return SolverPtr(new Solver(move(factorSkel), move(elimLumpRanges),
                                move(etTotalInvPerm),
                                blasOps()  // simpleOps()
                                ));
}

SolverPtr createSolverSchur(const Settings& settings,
                            const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss_,
                            const std::vector<uint64_t>& elimLumpRanges,
                            bool verbose) {
    BASPACHO_CHECK_GE(elimLumpRanges.size(), 2);
    SparseStructure ss =
        ss_.addIndependentEliminationFill(elimLumpRanges[0], elimLumpRanges[1]);
    for (uint64_t e = 1; e < elimLumpRanges.size() - 1; e++) {
        ss = ss.addIndependentEliminationFill(elimLumpRanges[e],
                                              elimLumpRanges[e + 1]);
    }

    uint64_t elimEnd = elimLumpRanges[elimLumpRanges.size() - 1];
    SparseStructure ssBottom = ss.extractRightBottom(elimEnd);

    // find best permutation for right-bottom corner that is left
    vector<uint64_t> permutation = ssBottom.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSsBottom =
        ssBottom.symmetricPermutation(invPerm, false);

    // apply permutation to param size of right-bottom corner
    std::vector<uint64_t> sortedBottomParamSize(paramSize.size() - elimEnd);
    for (size_t i = elimEnd; i < paramSize.size(); i++) {
        sortedBottomParamSize[invPerm[i - elimEnd]] = paramSize[i - elimEnd];
    }

    // compute as ordinary elimination tree on br-corner
    EliminationTree et(sortedBottomParamSize, sortedSsBottom);
    et.buildTree();
    et.computeMerges();
    et.computeAggregateStruct();

    // full sorted param size
    std::vector<uint64_t> sortedFullParamSize;
    sortedFullParamSize.reserve(paramSize.size());
    sortedFullParamSize.insert(sortedFullParamSize.begin(), paramSize.begin(),
                               paramSize.begin() + elimEnd);
    sortedFullParamSize.insert(sortedFullParamSize.begin(),
                               sortedBottomParamSize.begin(),
                               sortedBottomParamSize.end());

    BASPACHO_CHECK_EQ(et.spanStart.size() - 1,
                      et.lumpToSpan[et.lumpToSpan.size() - 1]);

    // compute span start as cumSum of `sortedFullParamSize`
    vector<uint64_t> fullSpanStart;
    fullSpanStart.reserve(sortedFullParamSize.size() + 1);
    fullSpanStart.insert(fullSpanStart.begin(), sortedFullParamSize.begin(),
                         sortedFullParamSize.end());
    fullSpanStart.push_back(0);
    cumSumVec(fullSpanStart);

    // compute lump to span, knowing up to elimEnd it's the identity
    vector<uint64_t> fullLumpToSpan(elimEnd + et.lumpToSpan.size());
    iota(fullLumpToSpan.begin(), fullLumpToSpan.begin() + elimEnd, 0);
    for (size_t i = 0; i < et.lumpToSpan.size(); i++) {
        fullLumpToSpan[i + elimEnd] = elimEnd + et.lumpToSpan[i];
    }
    BASPACHO_CHECK_EQ(fullSpanStart.size() - 1,
                      fullLumpToSpan[fullLumpToSpan.size() - 1]);

    // colStart are aggregate lump columns
    // ss last rows are to be permuted according to etTotalInvPerm
    vector<uint64_t> etTotalInvPerm =
        composePermutations(et.permInverse, invPerm);
    vector<uint64_t> fullInvPerm(elimEnd + etTotalInvPerm.size());
    iota(fullInvPerm.begin(), fullInvPerm.begin() + elimEnd, 0);
    for (size_t i = 0; i < etTotalInvPerm.size(); i++) {
        fullInvPerm[i + elimEnd] = elimEnd + etTotalInvPerm[i];
    }
    SparseStructure sortedSsT =
        ss.symmetricPermutation(fullInvPerm, false).transpose(false);

    // fullColStart joining sortedSsT.ptrs + elimEndDataPtr (shifted)
    vector<uint64_t> fullColStart;
    fullColStart.reserve(elimEnd + et.colStart.size());
    fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                        sortedSsT.ptrs.begin() + elimEnd);
    uint64_t elimEndDataPtr = sortedSsT.ptrs[elimEnd];
    for (size_t i = 0; i < et.colStart.size(); i++) {
        fullColStart.push_back(elimEndDataPtr + et.colStart[i]);
    }
    BASPACHO_CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

    // fullRowParam joining sortedSsT.inds and et.rowParam (moved)
    vector<uint64_t> fullRowParam;
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
    std::vector<uint64_t> elimLumpRangesArg = elimLumpRanges;
    if (settings.findSparseEliminationRanges) {
        while (true) {
            uint64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
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