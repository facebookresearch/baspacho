
#include "Solver.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <chrono>
#include <iostream>

#include "EliminationTree.h"
#include "TestingUtils.h"
#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

Solver::Solver(BlockMatrixSkel&& skel_, std::vector<uint64_t>&& elimLumps_,
               OpsPtr&& ops_)
    : skel(std::move(skel_)),
      elimLumps(std::move(elimLumps_)),
      ops(std::move(ops_)) {
    opMatrixSkel = ops->prepareMatrixSkel(skel);
    for (uint64_t l = 0; l + 1 < elimLumps.size(); l++) {
        opElimination.push_back(
            ops->prepareElimination(skel, elimLumps[l], elimLumps[l + 1]));
    }
}

void Solver::factorLump(double* data, uint64_t lump) const {
    uint64_t lumpStart = skel.lumpStart[lump];
    uint64_t lumpSize = skel.lumpStart[lump + 1] - lumpStart;
    uint64_t chainColBegin = skel.chainColPtr[lump];
    uint64_t diagBlockOffset = skel.chainData[chainColBegin];

    // compute lower diag cholesky dec on diagonal block
    ops->potrf(lumpSize, data + diagBlockOffset);

    uint64_t boardColBegin = skel.boardColPtr[lump];
    uint64_t boardColEnd = skel.boardColPtr[lump + 1];
    uint64_t belowDiagChainColOrd = skel.boardChainColOrd[boardColBegin + 1];
    uint64_t numColChains = skel.boardChainColOrd[boardColEnd - 1];
    uint64_t belowDiagOffset =
        skel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t numRowsBelowDiag =
        skel.chainRowsTillEnd[chainColBegin + numColChains - 1] -
        skel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    if (numRowsBelowDiag == 0) {
        return;
    }

    ops->trsm(lumpSize, numRowsBelowDiag, data + diagBlockOffset,
              data + belowDiagOffset);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<MatRMaj<double>, 0, OuterStride>;
using OuterStridedMatK = Eigen::Map<const MatRMaj<double>, 0, OuterStride>;

void Solver::assemble(double* data, uint64_t lump, uint64_t boardIndexInSN,
                      OpaqueData& ax) const {
    auto start = hrc::now();

    uint64_t chainColBegin = skel.chainColPtr[lump];

    uint64_t boardColBegin = skel.boardColPtr[lump];
    uint64_t boardColEnd = skel.boardColPtr[lump + 1];

    uint64_t targetLump = skel.boardRowLump[boardColBegin + boardIndexInSN];
    uint64_t belowDiagChainColOrd =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN];
    uint64_t rowDataEnd0 =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.boardChainColOrd[boardColEnd - 1];

    uint64_t belowDiagStart =
        skel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t startRowInSuperNode =
        skel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

    uint64_t targetLumpSize =
        skel.lumpStart[targetLump + 1] - skel.lumpStart[targetLump];

    uint64_t rectRowBegin = startRowInSuperNode;
    uint64_t dstStride = targetLumpSize;
    uint64_t srcColDataOffset = chainColBegin + belowDiagChainColOrd;
    uint64_t numBlockRows = rowDataEnd1 - belowDiagChainColOrd;
    uint64_t numBlockCols = rowDataEnd0 - belowDiagChainColOrd;
    ops->assemble(*opMatrixSkel, ax, data, rectRowBegin,
                  dstStride,         //
                  srcColDataOffset,  //
                  numBlockRows, numBlockCols);

    assembleLastCallTime = tdelta(hrc::now() - start).count();
    assembleCalls++;
    assembleTotTime += assembleLastCallTime;
    assembleMaxCallTime = std::max(assembleMaxCallTime, assembleLastCallTime);
}

void Solver::eliminateBoard(double* data, uint64_t lump,
                            uint64_t boardIndexInSN, OpaqueData& ax) const {
    uint64_t lumpSize = skel.lumpStart[lump + 1] - skel.lumpStart[lump];
    uint64_t chainColBegin = skel.chainColPtr[lump];

    uint64_t boardColBegin = skel.boardColPtr[lump];
    uint64_t boardColEnd = skel.boardColPtr[lump + 1];

    uint64_t belowDiagChainColOrd =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN];
    uint64_t rowDataEnd0 =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.boardChainColOrd[boardColEnd - 1];

    uint64_t belowDiagStart =
        skel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t rowStart =
        skel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];
    uint64_t numRowsSub =
        skel.chainRowsTillEnd[chainColBegin + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.chainRowsTillEnd[chainColBegin + rowDataEnd1 - 1] - rowStart;

    /*ctx.stride = numRowsSub;
    ctx.tempBuffer.resize(numRowsSub * numRowsFull);
    ops->gemm(numRowsSub, numRowsFull, lumpSize, data + belowDiagStart,
              data + belowDiagStart, ctx.tempBuffer.data());

    assemble(data, lump, boardIndexInSN, ctx);*/

    ops->gemmToTemp(ax, numRowsSub, numRowsFull, lumpSize,
                    data + belowDiagStart, data + belowDiagStart);
    assemble(data, lump, boardIndexInSN, ax);
}

void Solver::factor(double* data, bool verbose) const {
    for (uint64_t l = 0; l + 1 < elimLumps.size(); l++) {
        LOG_IF(INFO, verbose) << "Elim set: " << l << " (" << elimLumps[l]
                              << ".." << elimLumps[l + 1] << ")";
        ops->doElimination(*opMatrixSkel, data, elimLumps[l], elimLumps[l + 1],
                           *opElimination[l]);
    }

    uint64_t denseOpsFromLump =
        elimLumps.size() ? elimLumps[elimLumps.size() - 1] : 0;
    LOG_IF(INFO, verbose) << "Block-Fact from: " << denseOpsFromLump;

    double totPrepares = 0.0;
    OpaqueDataPtr ax = ops->createAssembleContext(*opMatrixSkel);
    for (uint64_t l = denseOpsFromLump; l < skel.chainColPtr.size() - 1; l++) {
        auto start = hrc::now();
        ops->prepareAssembleContext(*opMatrixSkel, *ax, l);
        totPrepares += tdelta(hrc::now() - start).count();

        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = skel.boardRowPtr[l],
                      rEnd = skel.boardRowPtr[l + 1];     //
             rPtr < rEnd && skel.boardColLump[rPtr] < l;  //
             rPtr++) {
            uint64_t origAggreg = skel.boardColLump[rPtr];
            if (origAggreg < denseOpsFromLump) {
                continue;
            }
            uint64_t boardIndexInSN = skel.boardColOrd[rPtr];
            uint64_t boardSNDataStart = skel.boardColPtr[origAggreg];
            uint64_t boardSNDataEnd = skel.boardColPtr[origAggreg + 1];
            CHECK_LT(boardIndexInSN, boardSNDataEnd - boardSNDataStart);
            CHECK_EQ(l, skel.boardRowLump[boardSNDataStart + boardIndexInSN]);
            eliminateBoard(data, origAggreg, boardIndexInSN, *ax);  // ctx);
        }

        factorLump(data, l);
    }

    LOG_IF(INFO, verbose) << "solver stats:"
                          << "\nprepares: " << totPrepares
                          << "\nassemble: #=" << assembleCalls
                          << ", time=" << assembleTotTime
                          << "s, last=" << assembleLastCallTime
                          << "s, max=" << assembleMaxCallTime << "s";
}

pair<uint64_t, bool> findLargestIndependentLumpSet(const BlockMatrixSkel& skel,
                                                   uint64_t startLump,
                                                   uint64_t maxSize = 8) {
    uint64_t limit = kInvalid;
    for (uint64_t a = startLump; a < skel.lumpToSpan.size() - 1; a++) {
        if (a >= limit) {
            break;
        }
        if (skel.lumpStart[a + 1] - skel.lumpStart[a] > maxSize) {
            return make_pair(a, true);
        }
        uint64_t aPtrStart = skel.boardColPtr[a];
        uint64_t aPtrEnd = skel.boardColPtr[a + 1];
        CHECK_EQ(skel.boardRowLump[aPtrStart], a);
        CHECK_LE(2, aPtrEnd - aPtrStart);
        limit = min(skel.boardRowLump[aPtrStart + 1], limit);
    }
    return make_pair(min(limit, skel.lumpToSpan.size()), false);
}

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
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

    BlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart, et.rowParam);

    // find progressive Schur elimination sets
    std::vector<uint64_t> elimLumpRanges{0};
    while (true) {
        uint64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
        auto [rangeEnd, hitSizeLimit] =
            findLargestIndependentLumpSet(skel, rangeStart);
        if (rangeEnd < rangeStart + 10) {
            break;
        }
        LOG_IF(INFO, verbose)
            << "Adding indep set: " << rangeStart << ".." << rangeEnd << endl;
        elimLumpRanges.push_back(rangeEnd);
        if (hitSizeLimit) {
            break;
        }
    }
    LOG_IF(INFO, verbose) << "Ranges: " << printVec(elimLumpRanges);
    if (elimLumpRanges.size() == 1) {
        elimLumpRanges.pop_back();
    }

    // LOG_IF(INFO, verbose) << "Lumps:\n" << printVec(et.lumpToSpan) <<
    // endl; LOG_IF(INFO, verbose) << "Largest indep set is 0.." <<
    // largestIndep << endl;

    return SolverPtr(new Solver(move(skel), move(elimLumpRanges),
                                blasOps()  // simpleOps()
                                ));
}

SolverPtr createSolverSchur(const std::vector<uint64_t>& paramSize,
                            const SparseStructure& ss_,
                            const std::vector<uint64_t>& elimLumps,
                            bool verbose) {
    CHECK_GE(elimLumps.size(), 2);
    SparseStructure ss =
        ss_.addIndependentEliminationFill(elimLumps[0], elimLumps[1]);
    for (uint64_t e = 1; e < elimLumps.size() - 1; e++) {
        ss = ss.addIndependentEliminationFill(elimLumps[e], elimLumps[e + 1]);
    }

    uint64_t elimEnd = elimLumps[elimLumps.size() - 1];
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

    CHECK_EQ(et.spanStart.size() - 1, et.lumpToSpan[et.lumpToSpan.size() - 1]);

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
    CHECK_EQ(fullSpanStart.size() - 1,
             fullLumpToSpan[fullLumpToSpan.size() - 1]);

    // colStart are aggregate lump columns
    // ss last rows are to be permuted according to etTotalPerm
    // TODO: optimize if slow
    vector<uint64_t> etTotalPerm = composePermutations(et.permInverse, invPerm);
    vector<uint64_t> fullInvPerm(elimEnd + etTotalPerm.size());
    iota(fullInvPerm.begin(), fullInvPerm.begin() + elimEnd, 0);
    for (size_t i = 0; i < etTotalPerm.size(); i++) {
        fullInvPerm[i + elimEnd] = elimEnd + etTotalPerm[i];
    }
    SparseStructure sortedSsT =
        ss.symmetricPermutation(fullInvPerm, false).transpose(true);

    vector<uint64_t> fullColStart;
    fullColStart.reserve(elimEnd + et.colStart.size());
    fullColStart.insert(fullColStart.begin(), sortedSsT.ptrs.begin(),
                        sortedSsT.ptrs.begin() + elimEnd);
    uint64_t elimEndDataPtr = sortedSsT.ptrs[elimEnd];
    for (size_t i = 0; i < et.colStart.size(); i++) {
        fullColStart.push_back(elimEndDataPtr + et.colStart[i]);
    }
    CHECK_EQ(fullColStart.size(), fullLumpToSpan.size());

    vector<uint64_t> fullRowParam;
    fullRowParam.reserve(elimEndDataPtr + et.rowParam.size());
    fullRowParam.insert(fullRowParam.begin(), sortedSsT.inds.begin(),
                        sortedSsT.inds.begin() + elimEndDataPtr);
    for (size_t i = 0; i < et.rowParam.size(); i++) {
        fullRowParam.push_back(et.rowParam[i] + elimEnd);
    }
    CHECK_EQ(fullRowParam.size(), fullColStart[fullColStart.size() - 1]);

    BlockMatrixSkel skel(fullSpanStart, fullLumpToSpan, fullColStart,
                         fullRowParam);

    // find (additional) progressive Schur elimination sets
    std::vector<uint64_t> elimLumpRanges = elimLumps;
    while (true) {
        uint64_t rangeStart = elimLumpRanges[elimLumpRanges.size() - 1];
        auto [rangeEnd, hitSizeLimit] =
            findLargestIndependentLumpSet(skel, rangeStart);
        if (rangeEnd < rangeStart + 10) {
            break;
        }
        LOG_IF(INFO, verbose)
            << "Adding indep set: " << rangeStart << ".." << rangeEnd << endl;
        elimLumpRanges.push_back(rangeEnd);
        if (hitSizeLimit) {
            break;
        }
    }
    LOG_IF(INFO, verbose) << "Ranges: " << printVec(elimLumpRanges);
    if (elimLumpRanges.size() == 1) {
        elimLumpRanges.pop_back();
    }

    return SolverPtr(new Solver(move(skel), move(elimLumpRanges),
                                blasOps()  // simpleOps()
                                ));
}