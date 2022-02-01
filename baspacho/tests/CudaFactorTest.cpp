#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/CoalescedBlockMatrix.h"
#include "baspacho/CudaDefs.h"
#include "baspacho/EliminationTree.h"
#include "baspacho/Solver.h"
#include "baspacho/SparseStructure.h"
#include "baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;
using namespace ::testing;

void testCoalescedFactor(OpsPtr&& ops) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel factorSkel(spanStart, lumpToSpan, groupedSs.ptrs,
                                        groupedSs.inds);

    vector<double> data(factorSkel.dataSize());
    iota(data.begin(), data.end(), 13);
    factorSkel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = factorSkel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    Solver solver(std::move(factorSkel), {}, {}, std::move(ops));

    // call factor with data on device
    double* dataGPU;
    cuCHECK(cudaMalloc((void**)&dataGPU, data.size() * sizeof(double)));
    cuCHECK(cudaMemcpy(dataGPU, data.data(), data.size() * sizeof(double),
                       cudaMemcpyHostToDevice));
    solver.factor(dataGPU);
    cuCHECK(cudaMemcpy(data.data(), dataGPU, data.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
    cuCHECK(cudaFree(dataGPU));

    cout << "Verif:\n" << verifyMat << endl;

    Eigen::MatrixXd computedMat = solver.factorSkel.densify(data);

    cout << "Cmptd:\n" << computedMat << endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(CudaFactor, CoalescedFactor) { testCoalescedFactor(cudaOps()); }

#if 0
void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.037, 57 + i);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<int64_t> permutation = ss.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

        vector<int64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(et.spanStart, et.lumpToSpan,
                                            et.colStart, et.rowParam);

        vector<double> data =
            randomData(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        Solver solver(std::move(factorSkel), {}, {}, genOps());
        solver.factor(data.data());
        Eigen::MatrixXd computedMat = solver.factorSkel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-5);
    }
}

TEST(CudaFactor, CoalescedFactor_Many) {
    testCoalescedFactor_Many([] { return cudaOps(); });
}
#endif