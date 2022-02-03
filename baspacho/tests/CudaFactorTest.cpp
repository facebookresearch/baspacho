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

    Eigen::MatrixXd computedMat = solver.factorSkel.densify(data);

    cout << "Verif:\n" << verifyMat << endl;
    cout << "Cmptd:\n" << computedMat << endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-7);
}

TEST(CudaFactor, CoalescedFactor) { testCoalescedFactor(cudaOps()); }

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
        et.computeMerges(/* compute sparse elim ranges = */ false);
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(
            et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

        vector<double> data =
            randomData(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, 0, factorSkel.order() * 1.5);

        Eigen::MatrixXd verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        Solver solver(std::move(factorSkel), {}, {}, genOps());

        // call factor with data on device
        double* dataGPU;
        cuCHECK(cudaMalloc((void**)&dataGPU, data.size() * sizeof(double)));
        cuCHECK(cudaMemcpy(dataGPU, data.data(), data.size() * sizeof(double),
                           cudaMemcpyHostToDevice));
        solver.factor(dataGPU);
        cuCHECK(cudaMemcpy(data.data(), dataGPU, data.size() * sizeof(double),
                           cudaMemcpyDeviceToHost));
        cuCHECK(cudaFree(dataGPU));

        Eigen::MatrixXd computedMat = solver.factorSkel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-7);
    }
}

TEST(CudaFactor, CoalescedFactor_Many) {
    testCoalescedFactor_Many([] { return cudaOps(); });
}

void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<int64_t> permutation = ss.fillReducingPermutation();
        vector<int64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;

        vector<int64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges(/* compute sparse elim ranges = */ true);
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel factorSkel(
            et.computeSpanStart(), et.lumpToSpan, et.colStart, et.rowParam);

        vector<double> data =
            randomData(factorSkel.dataSize(), -1.0, 1.0, 9 + i);
        factorSkel.damp(data, 0, factorSkel.order() * 1.5);

        Eigen::MatrixXd verifyMat = factorSkel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        /*int64_t largestIndep =
            findLargestIndependentLumpSet(factorSkel, 0).first;*/
        ASSERT_GE(et.sparseElimRanges.size(), 2);
        int64_t largestIndep = et.sparseElimRanges[1];
        Solver solver(move(factorSkel),  // {0, largestIndep},
                      move(et.sparseElimRanges), {}, genOps());

        NumericCtxPtr<double> numCtx =
            solver.symCtx->createNumericCtx<double>(0);

        // call doElimination with data on device
        double* dataGPU;
        cuCHECK(cudaMalloc((void**)&dataGPU, data.size() * sizeof(double)));
        cuCHECK(cudaMemcpy(dataGPU, data.data(), data.size() * sizeof(double),
                           cudaMemcpyHostToDevice));
        // solver.factor(dataGPU);
        numCtx->doElimination(*solver.elimCtxs[0], dataGPU, 0, largestIndep);
        cuCHECK(cudaMemcpy(data.data(), dataGPU, data.size() * sizeof(double),
                           cudaMemcpyDeviceToHost));
        cuCHECK(cudaFree(dataGPU));

        Eigen::MatrixXd computedMat = solver.factorSkel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .leftCols(largestIndep)
                .norm(),
            0, 1e-5);
    }
}

TEST(CudaFactor, SparseElim_Many_Blas) {
    testSparseElim_Many([] { return cudaOps(); });
}