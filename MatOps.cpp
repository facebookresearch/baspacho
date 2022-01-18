
#include "MatOps.h"

#include <glog/logging.h>

// simple ops implemented using Eigen (therefore single thread)
struct SimpleOps : Ops {
    // will just contain a reference to the skel
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
                                             uint64_t aggrEnd) override {
        return OpaqueDataPtr();
    }

    virtual void doElimination(const OpaqueData& ref, double* data,
                               uint64_t aggrStart, uint64_t aggrEnd,
                               const OpaqueData& elimData) override {
        const OpaqueDataMatrixSkel* pSkel =
            dynamic_cast<const OpaqueDataMatrixSkel*>(&ref);
        CHECK_NOTNULL(pSkel);
        const OpaqueDataMatrixSkel& skel = *pSkel;
    }

    virtual void potrf(uint64_t n, double* A) override {
        Eigen::Map<MatRMaj<double>> matA(A, n, n);
        Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(matA);
    }

    virtual void trsm(uint64_t n, uint64_t k, const double* A,
                      double* B) override {
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
                      const double* B, double* C) override {
        Eigen::Map<const MatRMaj<double>> matA(A, m, k);
        Eigen::Map<const MatRMaj<double>> matB(B, n, k);
        Eigen::Map<MatRMaj<double>> matC(C, n, m);
        matC = matB * matA.transpose();
    }

    // TODO
    // virtual void assemble();
};

OpsPtr simpleOps() { return OpsPtr(new SimpleOps); }
