
#include "MatOps.h"

#include <glog/logging.h>

#include <chrono>

#include "TestingUtils.h"
#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

static void factorAggreg(const BlockMatrixSkel& skel, double* data,
                         uint64_t aggreg) {
    uint64_t rangeStart = skel.rangeStart[aggreg];
    uint64_t aggregSize = skel.rangeStart[aggreg + 1] - rangeStart;
    uint64_t colStart = skel.sliceColPtr[aggreg];
    uint64_t dataPtr = skel.sliceData[colStart];

    // compute lower diag cholesky dec on diagonal block
    Eigen::Map<MatRMaj<double>> diagBlock(data + dataPtr, aggregSize,
                                          aggregSize);
    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(diagBlock); }
    uint64_t gatheredStart = skel.slabColPtr[aggreg];
    uint64_t gatheredEnd = skel.slabColPtr[aggreg + 1];
    uint64_t rowDataStart = skel.slabSliceColOrd[gatheredStart + 1];
    uint64_t rowDataEnd = skel.slabSliceColOrd[gatheredEnd - 1];
    uint64_t belowDiagStart = skel.sliceData[colStart + rowDataStart];
    uint64_t numRows = skel.sliceRowsTillEnd[colStart + rowDataEnd - 1] -
                       skel.sliceRowsTillEnd[colStart + rowDataStart - 1];

    Eigen::Map<MatRMaj<double>> belowDiagBlock(data + belowDiagStart, numRows,
                                               aggregSize);
    diagBlock.triangularView<Eigen::Lower>()
        .transpose()
        .solveInPlace<Eigen::OnTheRight>(belowDiagBlock);
}

static void prepareContextForTargetAggreg(
    const BlockMatrixSkel& skel, uint64_t targetAggreg,
    vector<uint64_t>& paramToSliceOffset) {
    paramToSliceOffset.assign(skel.spanStart.size() - 1, 999999);
    for (uint64_t i = skel.sliceColPtr[targetAggreg],
                  iEnd = skel.sliceColPtr[targetAggreg + 1];
         i < iEnd; i++) {
        paramToSliceOffset[skel.sliceRowSpan[i]] = skel.sliceData[i];
    }
}

using OuterStride = Eigen::OuterStride<>;
using OuterStridedMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
    // will just contain a reference to the skel
    struct OpaqueDataMatrixSkel : OpaqueData {
        OpaqueDataMatrixSkel(const BlockMatrixSkel& skel) : skel(skel) {}
        virtual ~OpaqueDataMatrixSkel() {}
        const BlockMatrixSkel& skel;
    };

    virtual void printStats() const override {
        LOG(INFO) << "matOp stats:"
                  << "\nBiggest dense block: " << potrfBiggestN
                  << "\npotrf: #=" << potrfCalls << ", time=" << potrfTotTime
                  << "s, last=" << potrfLastCallTime
                  << "s, max=" << potrfMaxCallTime << "s"
                  << "\ntrsm: #=" << trsmCalls << ", time=" << trsmTotTime
                  << "s, last=" << trsmLastCallTime
                  << "s, max=" << trsmMaxCallTime << "s"
                  << "\ngemm: #=" << gemmCalls << ", time=" << gemmTotTime
                  << "s, last=" << gemmLastCallTime
                  << "s, max=" << gemmMaxCallTime << "s";
    }

    virtual OpaqueDataPtr prepareMatrixSkel(
        const BlockMatrixSkel& skel) override {
        return OpaqueDataPtr(new OpaqueDataMatrixSkel(skel));
    }

    struct OpaqueDataElimData : OpaqueData {
        OpaqueDataElimData() {}
        virtual ~OpaqueDataElimData() {}
        vector<uint64_t> rowPtr;       // row data pointer
        vector<uint64_t> colRange;     // col-range
        vector<uint64_t> sliceColOrd;  //
    };

    // TODO: unit test
    virtual OpaqueDataPtr prepareElimination(const BlockMatrixSkel& skel,
                                             uint64_t aggrStart,
                                             uint64_t aggrEnd) override {
        OpaqueDataElimData* elim = new OpaqueDataElimData;

        uint64_t pIndexBegin = skel.rangeToSpan[aggrEnd];
        uint64_t nParams = skel.spanStart.size() - 1 - pIndexBegin;
        elim->rowPtr.assign(nParams + 1, 0);
        for (uint64_t a = aggrStart; a < aggrEnd; a++) {
            for (uint64_t i = skel.sliceColPtr[a],
                          iEnd = skel.sliceColPtr[a + 1];
                 i < iEnd; i++) {
                uint64_t pIndex = skel.sliceRowSpan[i];
                if (pIndex < pIndexBegin) {
                    continue;
                }
                uint64_t pRelIndex = pIndex - pIndexBegin;
                elim->rowPtr[pRelIndex]++;
            }
        }
        uint64_t totEls = cumSum(elim->rowPtr);
        elim->colRange.resize(totEls);
        elim->sliceColOrd.resize(totEls);
        for (uint64_t a = aggrStart; a < aggrEnd; a++) {
            for (uint64_t iStart = skel.sliceColPtr[a],
                          iEnd = skel.sliceColPtr[a + 1], i = iStart;
                 i < iEnd; i++) {
                uint64_t pIndex = skel.sliceRowSpan[i];
                if (pIndex < pIndexBegin) {
                    continue;
                }
                uint64_t pRelIndex = pIndex - pIndexBegin;
                CHECK_LT(elim->rowPtr[pRelIndex], totEls);
                elim->colRange[elim->rowPtr[pRelIndex]] = a;
                elim->sliceColOrd[elim->rowPtr[pRelIndex]] = i - iStart;

                CHECK_EQ(
                    skel.sliceRowSpan
                        [iStart + elim->sliceColOrd[elim->rowPtr[pRelIndex]]],
                    pIndex);

                elim->rowPtr[pRelIndex]++;
            }
        }
        rewind(elim->rowPtr);
        return OpaqueDataPtr(elim);
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t aggrStart, uint64_t aggrEnd,
                               const OpaqueData& elimData) override {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        const OpaqueDataElimData* pElim =
            dynamic_cast<const OpaqueDataElimData*>(&elimData);
        CHECK_NOTNULL(pSkel);
        CHECK_NOTNULL(pElim);
        const BlockMatrixSkel& skel = pSkel->skel;
        const OpaqueDataElimData& elim = *pElim;

        // TODO: parallel
        for (uint64_t a = aggrStart; a < aggrEnd; a++) {
            factorAggreg(skel, data, a);
        }

        // TODO: parallel2
        std::vector<double> tempBuffer;
        std::vector<uint64_t> paramToSliceOffset;
        uint64_t pIndexBegin = skel.rangeToSpan[aggrEnd];
        uint64_t pNum = skel.spanStart.size() - 1;
        for (uint64_t pRelIndex = 0; pRelIndex < elim.rowPtr.size() - 1;
             pRelIndex++) {
            uint64_t param = pRelIndex + pIndexBegin;
            uint64_t targetAggreg = skel.spanToRange[param];
            uint64_t targetAggregSize = skel.rangeStart[targetAggreg + 1] -
                                        skel.rangeStart[targetAggreg];
            uint64_t offsetInAggreg =
                skel.spanStart[param] - skel.rangeStart[targetAggreg];
            prepareContextForTargetAggreg(skel, targetAggreg,
                                          paramToSliceOffset);

            for (uint64_t i = elim.rowPtr[pRelIndex],
                          iEnd = elim.rowPtr[pRelIndex + 1];
                 i < iEnd; i++) {
                uint64_t aggreg = elim.colRange[i];
                uint64_t sliceIdx = elim.sliceColOrd[i];
                uint64_t ptrStart = skel.sliceColPtr[aggreg];
                uint64_t ptrEnd = skel.sliceColPtr[aggreg + 1];
                CHECK_EQ(skel.sliceRowSpan[ptrStart + sliceIdx], param);
                uint64_t nRowsBase =
                    skel.sliceRowsTillEnd[ptrStart + sliceIdx - 1];
                uint64_t nRowsEnd0 = skel.sliceRowsTillEnd[ptrStart + sliceIdx];
                uint64_t nRowsEnd = skel.sliceRowsTillEnd[ptrEnd - 1];
                uint64_t belowDiagOffset = skel.sliceData[ptrStart + sliceIdx];
                uint64_t numRowsSub = nRowsEnd0 - nRowsBase;
                uint64_t numRowsFull = nRowsEnd - nRowsBase;
                CHECK_EQ(numRowsSub,
                         skel.spanStart[param + 1] - skel.spanStart[param]);
                uint64_t aggregSize =
                    skel.rangeStart[aggreg + 1] - skel.rangeStart[aggreg];

                Eigen::Map<MatRMaj<double>> belowDiagBlockSub(
                    data + belowDiagOffset, numRowsSub, aggregSize);
                Eigen::Map<MatRMaj<double>> belowDiagBlockFull(
                    data + belowDiagOffset, numRowsFull, aggregSize);

                tempBuffer.resize(numRowsFull * numRowsSub);
                Eigen::Map<MatRMaj<double>> prod(tempBuffer.data(), numRowsFull,
                                                 numRowsSub);
                prod = belowDiagBlockFull * belowDiagBlockSub.transpose();

                // assemble
                for (uint64_t ptr = ptrStart + sliceIdx; ptr < ptrEnd; ptr++) {
                    uint64_t p = skel.sliceRowSpan[ptr];
                    uint64_t rowStart =
                        skel.sliceRowsTillEnd[ptr - 1] - nRowsBase;
                    uint64_t rowEnd = skel.sliceRowsTillEnd[ptr] - nRowsBase;
                    uint64_t rowSize = rowEnd - rowStart;
                    CHECK_EQ(rowEnd - rowStart,
                             skel.spanStart[p + 1] - skel.spanStart[p]);

                    double* targetData =
                        data + offsetInAggreg + paramToSliceOffset[p];

                    OuterStridedMatM targetBlock(targetData, rowEnd - rowStart,
                                                 numRowsSub,
                                                 OuterStride(targetAggregSize));
                    targetBlock -=
                        prod.block(rowStart, 0, rowEnd - rowStart, numRowsSub);
                }
            }
        }
    }

    virtual void potrf(uint64_t n, double* A) override {
        auto start = hrc::now();

        Eigen::Map<MatRMaj<double>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(matA);

        potrfLastCallTime = tdelta(hrc::now() - start).count();
        potrfCalls++;
        potrfTotTime += potrfLastCallTime;
        potrfMaxCallTime = std::max(potrfMaxCallTime, potrfLastCallTime);
        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
        auto start = hrc::now();

        using MatCMajD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::ColMajor>;

        // col-major's upper = (row-major's lower).transpose()
        Eigen::Map<const MatCMajD> matA(A, n, n);
        Eigen::Map<MatRMaj<double>> matB(B, k, n);
        matA.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(
            matB);

        trsmLastCallTime = tdelta(hrc::now() - start).count();
        trsmCalls++;
        trsmTotTime += trsmLastCallTime;
        trsmMaxCallTime = std::max(trsmMaxCallTime, trsmLastCallTime);
    }

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) override {
        auto start = hrc::now();

        Eigen::Map<const MatRMaj<double>> matA(A, m, k);
        Eigen::Map<const MatRMaj<double>> matB(B, n, k);
        Eigen::Map<MatRMaj<double>> matC(C, n, m);
        matC = matB * matA.transpose();

        gemmLastCallTime = tdelta(hrc::now() - start).count();
        gemmCalls++;
        gemmTotTime += gemmLastCallTime;
        gemmMaxCallTime = std::max(gemmMaxCallTime, gemmLastCallTime);
    }

    uint64_t potrfBiggestN = 0;
    uint64_t potrfCalls = 0;
    double potrfTotTime = 0.0;
    double potrfLastCallTime;
    double potrfMaxCallTime = 0.0;
    uint64_t trsmCalls = 0;
    double trsmTotTime = 0.0;
    double trsmLastCallTime;
    double trsmMaxCallTime = 0.0;
    uint64_t gemmCalls = 0;
    double gemmTotTime = 0.0;
    double gemmLastCallTime;
    double gemmMaxCallTime = 0.0;

    // TODO
    // virtual void assemble();
};

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }

// BLAS/LAPACK famously go without headers.
extern "C" {

#if 0
#define BLAS_INT long
#else
#define BLAS_INT int
#endif

void dpotrf_(char* uplo, BLAS_INT* n, double* A, BLAS_INT* lda, BLAS_INT* info);

void dtrsm_(char* side, char* uplo, char* transa, char* diag, BLAS_INT* m,
            BLAS_INT* n, double* alpha, double* A, BLAS_INT* lda, double* B,
            BLAS_INT* ldb);

void dgemm_(char* transa, char* transb, BLAS_INT* m, BLAS_INT* n, BLAS_INT* k,
            double* alpha, double* A, BLAS_INT* lda, double* B, BLAS_INT* ldb,
            double* beta, double* C, BLAS_INT* ldc);
}

struct BlasOps : SimpleOps {
    virtual void potrf(uint64_t n, double* A) override {
        auto start = hrc::now();

        char argUpLo = 'U';
        BLAS_INT argN = n;
        BLAS_INT argLdA = n;
        BLAS_INT info;

        dpotrf_(&argUpLo, &argN, A, &argLdA, &info);

        potrfLastCallTime = tdelta(hrc::now() - start).count();
        potrfCalls++;
        potrfTotTime += potrfLastCallTime;
        potrfMaxCallTime = std::max(potrfMaxCallTime, potrfLastCallTime);
        potrfBiggestN = std::max(potrfBiggestN, n);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
        auto start = hrc::now();

        char argSide = 'L';
        char argUpLo = 'U';
        char argTransA = 'C';
        char argDiag = 'N';
        BLAS_INT argM = n;
        BLAS_INT argN = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = n;
        BLAS_INT argLdB = n;

        dtrsm_(&argSide, &argUpLo, &argTransA, &argDiag, &argM, &argN,
               &argAlpha, (double*)A, &argLdA, B, &argLdB);

        trsmLastCallTime = tdelta(hrc::now() - start).count();
        trsmCalls++;
        trsmTotTime += trsmLastCallTime;
        trsmMaxCallTime = std::max(trsmMaxCallTime, trsmLastCallTime);
    }

    // C = A * B'
    virtual void gemm(uint64_t m, uint64_t n, uint64_t k, const double* A,
                      const double* B, double* C) override {
        auto start = hrc::now();

        char argTransA = 'C';
        char argTransB = 'N';
        BLAS_INT argM = m;
        BLAS_INT argN = n;
        BLAS_INT argK = k;
        double argAlpha = 1.0;
        BLAS_INT argLdA = k;
        BLAS_INT argLdB = k;
        double argBeta = 0.0;
        BLAS_INT argLdC = m;

        dgemm_(&argTransA, &argTransB, &argM, &argN, &argK, &argAlpha,
               (double*)A, &argLdA, (double*)B, &argLdB, &argBeta, C, &argLdC);

        gemmLastCallTime = tdelta(hrc::now() - start).count();
        gemmCalls++;
        gemmTotTime += gemmLastCallTime;
        gemmMaxCallTime = std::max(gemmMaxCallTime, gemmLastCallTime);
    }
};

OpsPtr blasOps() { return OpsPtr(new BlasOps); }