
#include "Solver.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>

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
        Eigen::Map<MatRMaj<double>> matC(C, m, n);
        matC = matA * matB.transpose();
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

void Solver::factorAggreg(double* data, uint64_t aggreg) {
    LOG(INFO) << "a: " << aggreg;
    uint64_t aggregStart = skel.aggregStart[aggreg];
    uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
    uint64_t colStart = skel.blockColDataPtr[aggreg];
    uint64_t diagBlockOff = skel.blockData[colStart];

    // compute lower diag cholesky dec on diagonal block
    LOG(INFO) << "d: " << data << ", ptr: " << diagBlockOff
              << ", asz: " << aggregSize;
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

    LOG(INFO) << "d: " << data << ", ptr: " << belowDiagOff
              << ", nrows: " << numRows;
    ops->trsm(aggregSize, numRows, data + diagBlockOff, data + belowDiagOff);
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

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

static void eliminateAggregItem(const BlockMatrixSkel& skel, double* data,
                                uint64_t aggreg, uint64_t rowItem) {
    uint64_t aggregStart = skel.aggregStart[aggreg];
    uint64_t aggregSize = skel.aggregStart[aggreg + 1] - aggregStart;
    uint64_t colStart = skel.blockColDataPtr[aggreg];

    uint64_t gatheredStart = skel.blockColGatheredDataPtr[aggreg];
    uint64_t gatheredEnd = skel.blockColGatheredDataPtr[aggreg + 1];
    uint64_t rowDataStart =
        skel.blockRowAggregParamPtr[gatheredStart + rowItem];
    uint64_t rowDataEnd0 =
        skel.blockRowAggregParamPtr[gatheredStart + rowItem + 1];
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

    MatRMaj<double> prod = belowDiagBlockFull * belowDiagBlockSub.transpose();

    for (uint64_t c = rowDataStart; c < rowDataEnd0; c++) {
        uint64_t cStart =
            skel.endBlockNumRowsAbove[colStart + c - 1] - rowStart;
        uint64_t cSize =
            skel.endBlockNumRowsAbove[colStart + c] - cStart - rowStart;
        uint64_t cParam = skel.blockRowParam[colStart + c];
        for (uint64_t r = rowDataStart; r < rowDataEnd1; r++) {
            uint64_t rStart =
                skel.endBlockNumRowsAbove[colStart + r - 1] - rowStart;
            uint64_t rSize =
                skel.endBlockNumRowsAbove[colStart + r] - rStart - rowStart;
            uint64_t rParam = skel.blockRowParam[colStart + r];
            auto [offset, stride] = findBlock(skel, cParam, rParam);
            OuterStridedMatM target(data + offset, rSize, cSize,
                                    OuterStride(stride));
            auto orig = prod.block(rStart, cStart, rSize, cSize);
            target -= orig;
        }
    }
}

void Solver::factor(double* data) {
    for (uint64_t a = 0; a < skel.blockColDataPtr.size() - 1; a++) {
        //  iterate over columns having a non-trivial a-block
        for (uint64_t rPtr = skel.slabRowPtr[a],
                      rEnd = skel.slabRowPtr[a + 1];       //
             rPtr < rEnd && skel.slabAggregInd[rPtr] < a;  //
             rPtr++) {
            uint64_t colAggreg = skel.slabAggregInd[rPtr];
            uint64_t colDataOff = skel.slabColInd[rPtr];
            uint64_t colDataStart = skel.blockColGatheredDataPtr[colAggreg];
            uint64_t colDataEnd = skel.blockColGatheredDataPtr[colAggreg + 1];
            CHECK_LT(colDataOff, colDataEnd - colDataStart);
            CHECK_EQ(a, skel.blockRowAggreg[colDataStart + colDataOff]);
            eliminateAggregItem(skel, data, colAggreg, colDataOff);
        }
        factorAggreg(data, a);
    }
}

SolverPtr createSolver(const std::vector<uint64_t>& paramSize,
                       const SparseStructure& ss) {}
