#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "BlockMatrix.h"
#include "EliminationTree.h"
#include "Solver.h"
#include "SparseStructure.h"
#include "TestingUtils.h"
#include "Utils.h"

using namespace std;

void testSolver(OpsPtr&& ops) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> rangeToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), rangeToSpan));
    BlockMatrixSkel skel(spanStart, rangeToSpan, groupedSs.ptrs,
                         groupedSs.inds);

    uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    skel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    Solver solver(std::move(skel), std::vector<uint64_t>{}, std::move(ops));
    solver.factor(data.data());

    Eigen::MatrixXd computedMat = solver.skel.densify(data);
    std::cout << "COMPUT:\n" << computedMat << std::endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Solver, Solver_Blas) { testSolver(blasOps()); }

TEST(Solver, Solver_Ref) { testSolver(simpleOps()); }

void testSolverXt(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.037, 57 + i);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<uint64_t> permutation = ss.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

        vector<uint64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        BlockMatrixSkel skel(et.spanStart, et.rangeToSpan, et.colStart,
                             et.rowParam);

        uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
        // std::cout << "VERIF:\n" << verifyMat << std::endl;

        Solver solver(std::move(skel), std::vector<uint64_t>{}, genOps());
        solver.factor(data.data());

        Eigen::MatrixXd computedMat = solver.skel.densify(data);
        // std::cout << "COMPUT:\n" << computedMat << std::endl;

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-5);
    }
}

TEST(Solver, SolverXt_Blas) {
    testSolverXt([] { return blasOps(); });
}

TEST(Solver, SolverXt_Ref) {
    testSolverXt([] { return simpleOps(); });
}

uint64_t findLargestIndependentAggregSet(const BlockMatrixSkel& skel);

void testSolverXtElim(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<uint64_t> permutation = ss.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;  //.symmetricPermutation(invPerm, false);

        vector<uint64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        BlockMatrixSkel skel(et.spanStart, et.rangeToSpan, et.colStart,
                             et.rowParam);

        uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
        // std::cout << "VERIF:\n" << verifyMat << std::endl;

        uint64_t largestIndep = findLargestIndependentAggregSet(skel);
        cout << "Largest indep set is 0.." << largestIndep
             << " (nAggregs: " << et.rangeToSpan.size() - 1 << ")"
             << "\naggregs:" << printVec(et.rangeToSpan) << endl;

        Solver solver(std::move(skel),
                      std::vector<uint64_t>{0, largestIndep},  //
                      genOps());
        solver.ops->doElimination(*solver.opMatrixSkel, data.data(), 0,
                                  largestIndep, *solver.opElimination[0]);
        // solver.factor(data.data());

        Eigen::MatrixXd computedMat = solver.skel.densify(data);
        // std::cout << "COMPUT:\n" << computedMat << std::endl;

        int endDenseSize = solver.skel.spanStart[largestIndep];

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .leftCols(largestIndep)
                .norm(),
            0, 1e-5);
    }
}

TEST(Solver, SolverXtElim_Blas) {
    testSolverXtElim([] { return blasOps(); });
}

TEST(Solver, SolverXtElim_Ref) {
    testSolverXtElim([] { return simpleOps(); });
}

void testSolverXtElFact(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<uint64_t> permutation = ss.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;  //.symmetricPermutation(invPerm, false);

        vector<uint64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        BlockMatrixSkel skel(et.spanStart, et.rangeToSpan, et.colStart,
                             et.rowParam);

        uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
        // std::cout << "VERIF:\n" << verifyMat << std::endl;

        uint64_t largestIndep = findLargestIndependentAggregSet(skel);
        cout << "Largest indep set is 0.." << largestIndep
             << " (nAggregs: " << et.rangeToSpan.size() - 1 << ")"
             << "\naggregs:" << printVec(et.rangeToSpan) << endl;

        Solver solver(std::move(skel),
                      std::vector<uint64_t>{0, largestIndep},  //
                      genOps());
        solver.factor(data.data());

        Eigen::MatrixXd computedMat = solver.skel.densify(data);
        // std::cout << "COMPUT:\n" << computedMat << std::endl;

        int endDenseSize = solver.skel.spanStart[largestIndep];

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-5);
    }
}

TEST(Solver, SolverXtElFact_Blas) {
    testSolverXtElFact([] { return blasOps(); });
}

TEST(Solver, SolverXtElFact_Ref) {
    testSolverXtElFact([] { return simpleOps(); });
}