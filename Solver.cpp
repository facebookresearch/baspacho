
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

Solver::Solver(BlockMatrixSkel&& skel_, std::vector<uint64_t>&& elimRanges_,
               OpsPtr&& ops_)
    : skel(std::move(skel_)),
      elimRanges(std::move(elimRanges_)),
      ops(std::move(ops_)) {
    opMatrixSkel = ops->prepareMatrixSkel(skel);
    for (uint64_t r = 0; r + 1 < elimRanges.size(); r++) {
        opElimination.push_back(
            ops->prepareElimination(skel, elimRanges[r], elimRanges[r + 1]));
    }
}

void Solver::factorAggreg(double* data, uint64_t range) const {
    uint64_t rangeStart = skel.rangeStart[range];
    uint64_t rangeSize = skel.rangeStart[range + 1] - rangeStart;
    uint64_t sliceColBegin = skel.sliceColPtr[range];
    uint64_t diagBlockOffset = skel.sliceData[sliceColBegin];

    // compute lower diag cholesky dec on diagonal block
    ops->potrf(rangeSize, data + diagBlockOffset);

    uint64_t slabColBegin = skel.slabColPtr[range];
    uint64_t slabColEnd = skel.slabColPtr[range + 1];
    uint64_t belowDiagSliceColOrd = skel.slabSliceColOrd[slabColBegin + 1];
    uint64_t numColSlices = skel.slabSliceColOrd[slabColEnd - 1];
    uint64_t belowDiagOffset =
        skel.sliceData[sliceColBegin + belowDiagSliceColOrd];
    uint64_t numRowsBelowDiag =
        skel.sliceRowsTillEnd[sliceColBegin + numColSlices - 1] -
        skel.sliceRowsTillEnd[sliceColBegin + belowDiagSliceColOrd - 1];
    if (numRowsBelowDiag == 0) {
        return;
    }

    ops->trsm(rangeSize, numRowsBelowDiag, data + diagBlockOffset,
              data + belowDiagOffset);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<MatRMaj<double>, 0, OuterStride>;
using OuterStridedMatK = Eigen::Map<const MatRMaj<double>, 0, OuterStride>;

void Solver::prepareContextForTargetAggreg(uint64_t targetAggreg,
                                           SolverContext& ctx) const {
    ctx.paramToSliceOffset.assign(skel.spanStart.size() - 1, 999999);
    for (uint64_t i = skel.sliceColPtr[targetAggreg],
                  iEnd = skel.sliceColPtr[targetAggreg + 1];
         i < iEnd; i++) {
        ctx.paramToSliceOffset[skel.sliceRowSpan[i]] = skel.sliceData[i];
    }
}

void Solver::assemble(double* data, uint64_t range, uint64_t slabIndexInSN,
                      SolverContext& ctx) const {
    auto start = hrc::now();

    uint64_t sliceColBegin = skel.sliceColPtr[range];

    uint64_t slabColBegin = skel.slabColPtr[range];
    uint64_t slabColEnd = skel.slabColPtr[range + 1];

    uint64_t targetAggreg = skel.slabRowRange[slabColBegin + slabIndexInSN];
    uint64_t belowDiagSliceColOrd =
        skel.slabSliceColOrd[slabColBegin + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.slabSliceColOrd[slabColBegin + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.slabSliceColOrd[slabColEnd - 1];

    uint64_t belowDiagStart =
        skel.sliceData[sliceColBegin + belowDiagSliceColOrd];
    uint64_t startRowInSuperNode =
        skel.sliceRowsTillEnd[sliceColBegin + belowDiagSliceColOrd - 1];

    uint64_t targetAggregSize =
        skel.rangeStart[targetAggreg + 1] - skel.rangeStart[targetAggreg];

    const double* matProduct = ctx.tempBuffer.data();

    uint64_t dstStride = targetAggregSize;
    uint64_t srcStride = ctx.stride;

    // TODO: multithread here
    for (uint64_t r = belowDiagSliceColOrd; r < rowDataEnd1; r++) {
        uint64_t rStart =
            skel.sliceRowsTillEnd[sliceColBegin + r - 1] - startRowInSuperNode;
        uint64_t rSize = skel.sliceRowsTillEnd[sliceColBegin + r] - rStart -
                         startRowInSuperNode;
        uint64_t rParam = skel.sliceRowSpan[sliceColBegin + r];
        uint64_t rOffset = ctx.paramToSliceOffset[rParam];
        const double* matRowPtr = matProduct + rStart * ctx.stride;

        uint64_t cEnd = std::min(rowDataEnd0, r + 1);
        for (uint64_t c = belowDiagSliceColOrd; c < cEnd; c++) {
            uint64_t cStart = skel.sliceRowsTillEnd[sliceColBegin + c - 1] -
                              startRowInSuperNode;
            uint64_t cSize = skel.sliceRowsTillEnd[sliceColBegin + c] - cStart -
                             startRowInSuperNode;
            uint64_t cParam = skel.sliceRowSpan[sliceColBegin + c];
            // CHECK_EQ(skel.spanToRange[cParam], targetAggreg);
            uint64_t offsetInAggreg =
                skel.spanStart[cParam] - skel.rangeStart[targetAggreg];
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

void Solver::eliminateAggregItem(double* data, uint64_t range,
                                 uint64_t slabIndexInSN,
                                 SolverContext& ctx) const {
    uint64_t rangeSize = skel.rangeStart[range + 1] - skel.rangeStart[range];
    uint64_t sliceColBegin = skel.sliceColPtr[range];

    uint64_t slabColBegin = skel.slabColPtr[range];
    uint64_t slabColEnd = skel.slabColPtr[range + 1];

    uint64_t belowDiagSliceColOrd =
        skel.slabSliceColOrd[slabColBegin + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.slabSliceColOrd[slabColBegin + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.slabSliceColOrd[slabColEnd - 1];

    uint64_t belowDiagStart =
        skel.sliceData[sliceColBegin + belowDiagSliceColOrd];
    uint64_t rowStart =
        skel.sliceRowsTillEnd[sliceColBegin + belowDiagSliceColOrd - 1];
    uint64_t numRowsSub =
        skel.sliceRowsTillEnd[sliceColBegin + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.sliceRowsTillEnd[sliceColBegin + rowDataEnd1 - 1] - rowStart;

    ctx.stride = numRowsSub;
    ctx.tempBuffer.resize(numRowsSub * numRowsFull);
    ops->gemm(numRowsSub, numRowsFull, rangeSize, data + belowDiagStart,
              data + belowDiagStart, ctx.tempBuffer.data());

    assemble(data, range, slabIndexInSN, ctx);
}

void Solver::factor(double* data) const {
    for (uint64_t r = 0; r + 1 < elimRanges.size(); r++) {
        LOG(INFO) << "Elim " << r;
        ops->doElimination(*opMatrixSkel, data, elimRanges[r],
                           elimRanges[r + 1], *opElimination[r]);
    }

    uint64_t denseOpsFromRange =
        elimRanges.size() ? elimRanges[elimRanges.size() - 1] : 0;
    LOG(INFO) << "FactFrom " << denseOpsFromRange;

    SolverContext ctx;
    for (uint64_t r = denseOpsFromRange; r < skel.sliceColPtr.size() - 1; r++) {
        prepareContextForTargetAggreg(r, ctx);

        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = skel.slabRowPtr[r],
                      rEnd = skel.slabRowPtr[r + 1];      //
             rPtr < rEnd && skel.slabColRange[rPtr] < r;  //
             rPtr++) {
            uint64_t origAggreg = skel.slabColRange[rPtr];
            if (origAggreg < denseOpsFromRange) {
                continue;
            }
            uint64_t slabIndexInSN = skel.slabColOrd[rPtr];
            uint64_t slabSNDataStart = skel.slabColPtr[origAggreg];
            uint64_t slabSNDataEnd = skel.slabColPtr[origAggreg + 1];
            CHECK_LT(slabIndexInSN, slabSNDataEnd - slabSNDataStart);
            CHECK_EQ(r, skel.slabRowRange[slabSNDataStart + slabIndexInSN]);
            eliminateAggregItem(data, origAggreg, slabIndexInSN, ctx);
        }

        factorAggreg(data, r);
    }

    LOG(INFO) << "solver stats:"
              << "\nassemble: #=" << assembleCalls
              << ", time=" << assembleTotTime
              << "s, last=" << assembleLastCallTime
              << "s, max=" << assembleMaxCallTime << "s";
}

uint64_t findLargestIndependentAggregSet(const BlockMatrixSkel& skel) {
    uint64_t limit = kInvalid;
    for (uint64_t a = 0; a < skel.rangeToSpan.size() - 1; a++) {
        if (a >= limit) {
            break;
        }
        uint64_t aPtrStart = skel.slabColPtr[a];
        uint64_t aPtrEnd = skel.slabColPtr[a + 1];
        CHECK_EQ(skel.slabRowRange[aPtrStart], a);
        CHECK_LE(2, aPtrEnd - aPtrStart);
        limit = std::min(skel.slabRowRange[aPtrStart + 1], limit);
    }
    return std::min(limit, skel.rangeToSpan.size());
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

    // LOG(INFO) << "\naggregs: " << printVec(et.rangeToSpan);

    BlockMatrixSkel skel(et.spanStart, et.rangeToSpan, et.colStart,
                         et.rowParam);

    uint64_t largestIndep = findLargestIndependentAggregSet(skel);
    LOG(INFO) << "Largest indep set is 0.." << largestIndep;

    return SolverPtr(new Solver(std::move(skel),
                                std::vector<uint64_t>{0, largestIndep},
                                blasOps()  // simpleOps()
                                ));
}
