
#include "Solver.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

#include "TestingUtils.h"
#include "Utils.h"

using namespace std;

struct SimpleOps : Ops {
    struct OpaqueDataMatrixSkel : OpaqueData {
        OpaqueDataMatrixSkel(const BlockMatrixSkel& skel) : skel(skel) {}
        virtual ~OpaqueDataMatrixSkel() {}
        const BlockMatrixSkel& skel;
    };

    virtual OpaqueDataPtr prepareMatrixSkel(
        const BlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new OpaqueDataMatrixSkel(skel));
    }

    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t aggrStart,
                                             uint64_t aggrEnd) {
        return OpaqueDataPtr();
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t aggrStart, uint64_t aggrEnd,
                               const OpaqueData& elimData) {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        CHECK_NOTNULL(pSkel);
        const OpaqueDataMatrixSkel& skel = *pSkel;
    }

    virtual void potrf(uint64_t n, double* A) {
        Eigen::Map<MatRMaj<double>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(matA);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A, double* B) {
        using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        Eigen::Map<MatRMaj<double>> matB(B, k, n);
        matA.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(
            matB);
    }

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) {
        Eigen::Map<const MatRMaj<double>> matA(A, m, k);
        Eigen::Map<const MatRMaj<double>> matB(B, n, k);
        Eigen::Map<MatRMaj<double>> matC(C, n, m);
        matC = matB * matA.transpose();
    }

    // TODO
    // virtual void assemble();
};

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

Solver::Solver(BlockMatrixSkel&& skel, std::vector<uint64_t>&& elimRanges,
               OpsPtr ops)
    : skel(std::move(skel)),
      elimRanges(std::move(elimRanges)),
      ops(std::move(ops)) {}

void Solver::factorAggreg(double* data, uint64_t aggreg) const {
    uint64_t aggregStart = skel.aggregStart[aggreg];
    uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
    uint64_t colStart = skel.blockColDataPtr[aggreg];
    uint64_t diagBlockOff = skel.blockData[colStart];

    // compute lower diag cholesky dec on diagonal block
    ops->potrf(aggregSize, data + diagBlockOff);

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];
    uint64_t rowDataStart = skel.blockRowAggregParamPtr[gatheredStart + 1];
    uint64_t rowDataEnd = skel.blockRowAggregParamPtr[gatheredEnd - 1];
    uint64_t belowDiagOff = skel.blockData[colStart + rowDataStart];
    uint64_t numRows = skel.endBlockNumRowsAbove[colStart + rowDataEnd - 1] -
                       skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];
    if (numRows == 0) {
        return;
    }

    ops->trsm(aggregSize, numRows, data + diagBlockOff, data + belowDiagOff);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<MatRMaj<double>, 0, OuterStride>;
using OuterStridedMatK = Eigen::Map<const MatRMaj<double>, 0, OuterStride>;

static uint64_t bisect(const uint64_t* array, uint64_t size, uint64_t needle) {
    uint64_t a = 0, b = size;
    while (b - a > 1) {
        uint64_t m = (a + b) / 2;
        if (needle >= array[m]) {
            a = m;
        } else {
            b = m;
        }
    }
    return a;
}

// returns (offset, stride)
static std::pair<uint64_t, uint64_t> findBlock(const BlockMatrixSkel& skel,
                                               uint64_t cParam,
                                               uint64_t rParam) {
    uint64_t aggreg = skel.paramToAggreg[cParam];
    uint64_t aggregSize =
        skel.aggregStart[aggreg + 1] - skel.aggregStart[aggreg];
    uint64_t offsetInAggreg =
        skel.paramStart[cParam] - skel.aggregStart[aggreg];
    uint64_t start = skel.blockColDataPtr[aggreg];
    uint64_t end = skel.blockColDataPtr[aggreg + 1];
    // bisect to find rParam in blockRowParam[start:end]
    uint64_t pos =
        bisect(skel.blockRowParam.data() + start, end - start, rParam);
    CHECK_EQ(skel.blockRowParam[start + pos], rParam);
    return std::make_pair(skel.blockData[start + pos] + offsetInAggreg,
                          aggregSize);
}

static void assembleOld(const BlockMatrixSkel& skel, double* data,
                        const MatRMaj<double>& prod, uint64_t aggreg,
                        uint64_t slabIndexInSN) {
    uint64_t aggregSize =
        skel.aggregStart[aggreg + 1] - skel.aggregStart[aggreg];
    uint64_t colStart = skel.blockColDataPtr[aggreg];

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];

    uint64_t rowDataStart =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.blockRowAggregParamPtr[gatheredEnd - 1];

    uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
    uint64_t startRowInSuperNode =
        skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];

    for (uint64_t c = rowDataStart; c < rowDataEnd0; c++) {
        uint64_t cStart =
            skel.endBlockNumRowsAbove[colStart + c - 1] - startRowInSuperNode;
        uint64_t cSize = skel.endBlockNumRowsAbove[colStart + c] - cStart -
                         startRowInSuperNode;
        uint64_t cParam = skel.blockRowParam[colStart + c];
        for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
            uint64_t rStart = skel.endBlockNumRowsAbove[colStart + r - 1] -
                              startRowInSuperNode;
            uint64_t rSize = skel.endBlockNumRowsAbove[colStart + r] - rStart -
                             startRowInSuperNode;
            uint64_t rParam = skel.blockRowParam[colStart + r];
            auto [offset, stride] = findBlock(skel, cParam, rParam);
            OuterStridedMatM target(data + offset, rSize, cSize,
                                    OuterStride(stride));
            auto orig = prod.block(rStart, cStart, rSize, cSize);
            target -= orig;
        }
    }
}

void Solver::prepareContextForTargetAggreg(uint64_t targetAggreg,
                                           SolverContext& ctx) const {
    ctx.paramToSliceOffset.assign(skel.paramStart.size() - 1, 999999);
    for (uint64_t i = skel.blockColDataPtr[targetAggreg],
                  iEnd = skel.blockColDataPtr[targetAggreg + 1];
         i < iEnd; i++) {
        ctx.paramToSliceOffset[skel.blockRowParam[i]] = skel.blockData[i];
    }
}

void Solver::assemble(double* data, uint64_t aggreg, uint64_t slabIndexInSN,
                      SolverContext& ctx) const {
    uint64_t aggregSize =
        skel.aggregStart[aggreg + 1] - skel.aggregStart[aggreg];
    uint64_t colStart = skel.blockColDataPtr[aggreg];

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];

    uint64_t targetAggreg = skel.blockRowAggreg[gatheredStart + slabIndexInSN];
    uint64_t rowDataStart =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.blockRowAggregParamPtr[gatheredEnd - 1];

    uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
    uint64_t startRowInSuperNode =
        skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];

    uint64_t targetAggregSize =
        skel.aggregStart[targetAggreg + 1] - skel.aggregStart[targetAggreg];

    const double* matProduct = ctx.tempBuffer.data();

    for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
        uint64_t rStart =
            skel.endBlockNumRowsAbove[colStart + r - 1] - startRowInSuperNode;
        uint64_t rSize = skel.endBlockNumRowsAbove[colStart + r] - rStart -
                         startRowInSuperNode;
        uint64_t rParam = skel.blockRowParam[colStart + r];
        uint64_t rOffset = ctx.paramToSliceOffset[rParam];
        const double* matRowPtr = matProduct + rStart * ctx.stride;

        for (uint64_t c = rowDataStart; c < rowDataEnd0; c++) {
            uint64_t cStart = skel.endBlockNumRowsAbove[colStart + c - 1] -
                              startRowInSuperNode;
            uint64_t cSize = skel.endBlockNumRowsAbove[colStart + c] - cStart -
                             startRowInSuperNode;
            uint64_t cParam = skel.blockRowParam[colStart + c];
            CHECK_EQ(skel.paramToAggreg[cParam], targetAggreg);
            uint64_t offsetInAggreg =
                skel.paramStart[cParam] - skel.aggregStart[targetAggreg];
            uint64_t offset = rOffset + offsetInAggreg;

            OuterStridedMatM target(data + offset, rSize, cSize,
                                    OuterStride(targetAggregSize));
            target -= OuterStridedMatK(matRowPtr + cStart, rSize, cSize,
                                       OuterStride(ctx.stride));
        }
    }
}

void Solver::eliminateAggregItem(double* data, uint64_t aggreg,
                                 uint64_t slabIndexInSN,
                                 SolverContext& ctx) const {
    uint64_t aggregSize =
        skel.aggregStart[aggreg + 1] - skel.aggregStart[aggreg];
    uint64_t colStart = skel.blockColDataPtr[aggreg];

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];

    uint64_t rowDataStart =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN];
    uint64_t rowDataEnd0 =
        skel.blockRowAggregParamPtr[gatheredStart + slabIndexInSN + 1];
    uint64_t rowDataEnd1 = skel.blockRowAggregParamPtr[gatheredEnd - 1];

    uint64_t belowDiagStart = skel.blockData[colStart + rowDataStart];
    uint64_t rowStart = skel.endBlockNumRowsAbove[colStart + rowDataStart - 1];
    uint64_t numRowsSub =
        skel.endBlockNumRowsAbove[colStart + rowDataEnd0 - 1] - rowStart;
    uint64_t numRowsFull =
        skel.endBlockNumRowsAbove[colStart + rowDataEnd1 - 1] - rowStart;

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
    for (uint64_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {
        prepareContextForTargetAggreg(a, ctx);

        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = skel.slabRowPtr[a],
                      rEnd = skel.slabRowPtr[a + 1];       //
             rPtr < rEnd && skel.slabAggregInd[rPtr] < a;  //
             rPtr++) {
            uint64_t origAggreg = skel.slabAggregInd[rPtr];
            uint64_t slabIndexInSN = skel.slabColInd[rPtr];
            uint64_t slabSNDataStart = skel.blockColGatheredDataPtr[origAggreg];
            uint64_t slabSNDataEnd =
                skel.blockColGatheredDataPtr[origAggreg + 1];
            CHECK_LT(slabIndexInSN, slabSNDataEnd - slabSNDataStart);
            CHECK_EQ(a, skel.blockRowAggreg[slabSNDataStart + slabIndexInSN]);
            eliminateAggregItem(data, origAggreg, slabIndexInSN, ctx);
        }
        factorAggreg(data, a);
    }
}

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss) {}
