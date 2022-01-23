
#include "Solver.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <chrono>

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

void Solver::prepareContextForTargetAggreg(uint64_t targetAggreg,
                                           SolverContext& ctx) const {
    ctx.paramToChainOffset.assign(skel.spanStart.size() - 1, 999999);
    for (uint64_t i = skel.chainColPtr[targetAggreg],
                  iEnd = skel.chainColPtr[targetAggreg + 1];
         i < iEnd; i++) {
        ctx.paramToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }
}

void Solver::assemble(double* data, uint64_t lump, uint64_t boardIndexInSN,
                      SolverContext& ctx) const {
    auto start = hrc::now();

    uint64_t chainColBegin = skel.chainColPtr[lump];

    uint64_t boardColBegin = skel.boardColPtr[lump];
    uint64_t boardColEnd = skel.boardColPtr[lump + 1];

    uint64_t targetAggreg = skel.boardRowLump[boardColBegin + boardIndexInSN];
    uint64_t belowDiagChainColOrd =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN];
    uint64_t rowDataEnd0 =
        skel.boardChainColOrd[boardColBegin + boardIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.boardChainColOrd[boardColEnd - 1];

    uint64_t belowDiagStart =
        skel.chainData[chainColBegin + belowDiagChainColOrd];
    uint64_t startRowInSuperNode =
        skel.chainRowsTillEnd[chainColBegin + belowDiagChainColOrd - 1];

    uint64_t targetAggregSize =
        skel.lumpStart[targetAggreg + 1] - skel.lumpStart[targetAggreg];

    const double* matProduct = ctx.tempBuffer.data();

    uint64_t dstStride = targetAggregSize;
    uint64_t srcStride = ctx.stride;

    // TODO: multithread here
    for (uint64_t r = belowDiagChainColOrd; r < rowDataEnd1; r++) {
        uint64_t rBegin =
            skel.chainRowsTillEnd[chainColBegin + r - 1] - startRowInSuperNode;
        uint64_t rSize = skel.chainRowsTillEnd[chainColBegin + r] - rBegin -
                         startRowInSuperNode;
        uint64_t rParam = skel.chainRowSpan[chainColBegin + r];
        uint64_t rOffset = ctx.paramToChainOffset[rParam];
        const double* matRowPtr = matProduct + rBegin * ctx.stride;

        uint64_t cEnd = std::min(rowDataEnd0, r + 1);
        for (uint64_t c = belowDiagChainColOrd; c < cEnd; c++) {
            uint64_t cStart = skel.chainRowsTillEnd[chainColBegin + c - 1] -
                              startRowInSuperNode;
            uint64_t cSize = skel.chainRowsTillEnd[chainColBegin + c] - cStart -
                             startRowInSuperNode;
            uint64_t cParam = skel.chainRowSpan[chainColBegin + c];
            // CHECK_EQ(skel.spanToLump[cParam], targetAggreg);
            uint64_t offsetInAggreg =
                skel.spanStart[cParam] - skel.lumpStart[targetAggreg];
            uint64_t offset = rOffset + offsetInAggreg;

            // TODO: investigate why is Eigen MUCH slower here?!
            double* dst = data + offset;
            const double* src = matRowPtr + cStart;
            for (uint j = 0; j < rSize; j++) {
                for (uint i = 0; i < cSize; i++) {
                    dst[i] -= src[i];
                }
                dst += dstStride;
                src += srcStride;
            }
#if 0
            OuterStridedMatM target(data + offset, rSize, cSize,
                                    OuterStride(targetAggregSize));
            target -= OuterStridedMatK(matRowPtr + cStart, rSize, cSize,
                                       OuterStride(ctx.stride));
#endif
        }
    }

    assembleLastCallTime = tdelta(hrc::now() - start).count();
    assembleCalls++;
    assembleTotTime += assembleLastCallTime;
    assembleMaxCallTime = std::max(assembleMaxCallTime, assembleLastCallTime);
}

void Solver::eliminateBoard(double* data, uint64_t lump,
                            uint64_t boardIndexInSN, SolverContext& ctx) const {
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

    ctx.stride = numRowsSub;
    ctx.tempBuffer.resize(numRowsSub * numRowsFull);
    ops->gemm(numRowsSub, numRowsFull, lumpSize, data + belowDiagStart,
              data + belowDiagStart, ctx.tempBuffer.data());

    assemble(data, lump, boardIndexInSN, ctx);
}

void Solver::factor(double* data) const {
    for (uint64_t l = 0; l + 1 < elimLumps.size(); l++) {
        LOG(INFO) << "Elim " << l;
        ops->doElimination(*opMatrixSkel, data, elimLumps[l], elimLumps[l + 1],
                           *opElimination[l]);
    }

    uint64_t denseOpsFromLump =
        elimLumps.size() ? elimLumps[elimLumps.size() - 1] : 0;
    LOG(INFO) << "FactFrom " << denseOpsFromLump;

    SolverContext ctx;
    for (uint64_t l = denseOpsFromLump; l < skel.chainColPtr.size() - 1; l++) {
        prepareContextForTargetAggreg(l, ctx);

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
            eliminateBoard(data, origAggreg, boardIndexInSN, ctx);
        }

        factorLump(data, l);
    }

    LOG(INFO) << "solver stats:"
              << "\nassemble: #=" << assembleCalls
              << ", time=" << assembleTotTime
              << "s, last=" << assembleLastCallTime
              << "s, max=" << assembleMaxCallTime << "s";
}

uint64_t findLargestIndependentAggregSet(const BlockMatrixSkel& skel) {
    uint64_t limit = kInvalid;
    for (uint64_t a = 0; a < skel.lumpToSpan.size() - 1; a++) {
        if (a >= limit) {
            break;
        }
        uint64_t aPtrStart = skel.boardColPtr[a];
        uint64_t aPtrEnd = skel.boardColPtr[a + 1];
        CHECK_EQ(skel.boardRowLump[aPtrStart], a);
        CHECK_LE(2, aPtrEnd - aPtrStart);
        limit = std::min(skel.boardRowLump[aPtrStart + 1], limit);
    }
    return std::min(limit, skel.lumpToSpan.size());
}

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss) {
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

    // LOG(INFO) << "\naggregs: " << printVec(et.lumpToSpan);

    BlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart, et.rowParam);

    uint64_t largestIndep = findLargestIndependentAggregSet(skel);
    LOG(INFO) << "Largest indep set is 0.." << largestIndep;

    return SolverPtr(new Solver(std::move(skel),
                                std::vector<uint64_t>{0, largestIndep},
                                blasOps()  // simpleOps()
                                ));
}
