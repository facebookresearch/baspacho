
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
    LOG(INFO) << "wth? " << elimRanges.size();
    for (uint64_t r = 0; r + 1 < elimRanges.size(); r++) {
        // ops->prepareElimination(skel, elimRanges[r], elimRanges[r + 1]);
        opElimination.push_back(
            ops->prepareElimination(skel, elimRanges[r], elimRanges[r + 1]));
    }
}

void Solver::factorAggreg(double* data, uint64_t aggreg) const {
    uint64_t rangeStart = skel.rangeStart[aggreg];
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - rangeStart;
    uint64_t colStart = skel.sliceColPtr[aggreg];
    uint64_t diagBlockOff = skel.sliceData[colStart];

    // compute lower diag cholesky dec on diagonal block
    ops->potrf(aggregSize, data + diagBlockOff);

    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.slabSliceColOrd[gatheredEnd - 1];
    uint64_t belowDiagOff = skel.sliceData[colStart + rowDataStart];
    uint64_t numRows = skel.sliceRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.sliceRowsTillEnd[colStart + rowDataStart - 1];
    if (numRows == 0) {
        return;
    }

    ops->trsm(aggregSize, numRows, data + diagBlockOff, data + belowDiagOff);
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

void Solver::assemble(double* data, uint64_t aggreg, uint64_t slabIndexInSN,
                      SolverContext& ctx) const {
    auto start = hrc::now();

    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - skel.rangeStart[aggreg];
    uint64_t colStart = skel.sliceColPtr[aggreg];

    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];

    uint64_t targetAggreg = skel.slabRowRange[gatheredStart + slabIndexInSN];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.slabSliceColOrd[gatheredStart + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.slabSliceColOrd[gatheredEnd - 1];

    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t startRowInSuperNode =
        skel.sliceRowsTillEnd[colStart + rowDataStart - 1];

    uint64_t targetAggregSize =
        skel.rangeStart[targetAggreg + 1] - skel.rangeStart[targetAggreg];

    const double* matProduct = ctx.tempBuffer.data();

    uint64_t dstStride = targetAggregSize;
    uint64_t srcStride = ctx.stride;

    // TODO: multithread here
    for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
        uint64_t rStart =
            skel.sliceRowsTillEnd[colStart + r - 1] - startRowInSuperNode;
        uint64_t rSize =
            skel.sliceRowsTillEnd[colStart + r] - rStart - startRowInSuperNode;
        uint64_t rParam = skel.sliceRowSpan[colStart + r];
        uint64_t rOffset = ctx.paramToSliceOffset[rParam];
        const double* matRowPtr = matProduct + rStart * ctx.stride;

        uint64_t cEnd = std::min(rowDataEnd0, r + 1);
        for (uint64_t c = rowDataStart; c < cEnd; c++) {
            uint64_t cStart =
                skel.sliceRowsTillEnd[colStart + c - 1] - startRowInSuperNode;
            uint64_t cSize = skel.sliceRowsTillEnd[colStart + c] - cStart -
                             startRowInSuperNode;
            uint64_t cParam = skel.sliceRowSpan[colStart + c];
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

void Solver::eliminateAggregItem(double* data, uint64_t aggreg,
                                 uint64_t slabIndexInSN,
                                 SolverContext& ctx) const {
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - skel.rangeStart[aggreg];
    uint64_t colStart = skel.sliceColPtr[aggreg];

    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];

    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.slabSliceColOrd[gatheredStart + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.slabSliceColOrd[gatheredEnd - 1];

    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t rowStart = skel.sliceRowsTillEnd[colStart + rowDataStart - 1];
    uint64_t numRowsSub =
        skel.sliceRowsTillEnd[colStart + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.sliceRowsTillEnd[colStart + rowDataEnd1 - 1] - rowStart;

    Eigen::Map<MatRMaj<double>> belowDiagBlockSub(data + belowDiagStart,
                                                  numRowsSub, aggregSize);
    Eigen::Map<MatRMaj<double>> belowDiagBlockFull(data + belowDiagStart,
                                                   numRowsFull, aggregSize);

    ctx.stride = numRowsSub;
    ctx.tempBuffer.resize(numRowsSub * numRowsFull);
    ops->gemm(numRowsSub, numRowsFull, aggregSize, data + belowDiagStart,
              data + belowDiagStart, ctx.tempBuffer.data());

    assemble(data, aggreg, slabIndexInSN, ctx);
}

void Solver::factor(double* data) const {
    SolverContext ctx;
    for (uint64_t a = 0; a < skel.sliceColPtr.size() - 1; a++) {
        prepareContextForTargetAggreg(a, ctx);

        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = skel.slabRowPtr[a],
                      rEnd = skel.slabRowPtr[a + 1];      //
             rPtr < rEnd && skel.slabColRange[rPtr] < a;  //
             rPtr++) {
            uint64_t origAggreg = skel.slabColRange[rPtr];
            uint64_t slabIndexInSN = skel.slabColOrd[rPtr];
            uint64_t slabSNDataStart = skel.slabColPtr[origAggreg];
            uint64_t slabSNDataEnd = skel.slabColPtr[origAggreg + 1];
            CHECK_LT(slabIndexInSN, slabSNDataEnd - slabSNDataStart);
            CHECK_EQ(a, skel.slabRowRange[slabSNDataStart + slabIndexInSN]);
            eliminateAggregItem(data, origAggreg, slabIndexInSN, ctx);
        }
        factorAggreg(data, a);
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

    return SolverPtr(new Solver(std::move(skel), std::vector<uint64_t>{},
                                blasOps()  // simpleOps()
                                ));
}
