#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "../../testing/TestingUtils.h"
#include "../CoalescedBlockMatrix.h"
#include "../EliminationTree.h"
#include "../Solver.h"
#include "../SparseStructure.h"
#include "../Utils.h"

using namespace std;

void testCoalescedFactor(OpsPtr&& ops) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);

    uint64_t totData = skel.chainData[skel.chainData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    skel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    Solver solver(std::move(skel), std::vector<uint64_t>{}, std::move(ops));
    solver.factor(data.data());
    Eigen::MatrixXd computedMat = solver.skel.densify(data);

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Factor, CoalescedFactor_Blas) { testCoalescedFactor(blasOps()); }

TEST(Factor, CoalescedFactor_Ref) { testCoalescedFactor(simpleOps()); }

void testCoalescedFactor_Many(const std::function<OpsPtr()>& genOps) {
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

        CoalescedBlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart,
                                      et.rowParam);

        uint64_t totData = skel.chainData[skel.chainData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        Solver solver(std::move(skel), std::vector<uint64_t>{}, genOps());
        solver.factor(data.data());
        Eigen::MatrixXd computedMat = solver.skel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-5);
    }
}

TEST(Factor, CoalescedFactor_Many_Blas) {
    testCoalescedFactor_Many([] { return blasOps(); });
}

TEST(Factor, CoalescedFactor_Many_Ref) {
    testCoalescedFactor_Many([] { return simpleOps(); });
}

pair<uint64_t, bool> findLargestIndependentLumpSet(
    const CoalescedBlockMatrixSkel& skel, uint64_t startLump,
    uint64_t maxSize = 8);

void testSparseElim_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<uint64_t> permutation = ss.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;

        vector<uint64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart,
                                      et.rowParam);

        uint64_t totData = skel.chainData[skel.chainData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        uint64_t largestIndep = findLargestIndependentLumpSet(skel, 0).first;
        Solver solver(std::move(skel),
                      std::vector<uint64_t>{0, largestIndep},  //
                      genOps());
        solver.ops->doElimination(*solver.opMatrixSkel, data.data(), 0,
                                  largestIndep, *solver.opElimination[0]);
        Eigen::MatrixXd computedMat = solver.skel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .leftCols(largestIndep)
                .norm(),
            0, 1e-5);
    }
}

TEST(Factor, SparseElim_Many_Blas) {
    testSparseElim_Many([] { return blasOps(); });
}

TEST(Factor, SparseElim_Many_Ref) {
    testSparseElim_Many([] { return simpleOps(); });
}

void testSparseElimAndFactor_Many(const std::function<OpsPtr()>& genOps) {
    for (int i = 0; i < 20; i++) {
        auto colBlocks = randomCols(115, 0.03, 57 + i);
        colBlocks = makeIndependentElimSet(colBlocks, 0, 60);
        SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

        vector<uint64_t> permutation = ss.fillReducingPermutation();
        vector<uint64_t> invPerm = inversePermutation(permutation);
        SparseStructure sortedSs = ss;

        vector<uint64_t> paramSize =
            randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
        EliminationTree et(paramSize, sortedSs);
        et.buildTree();
        et.computeMerges();
        et.computeAggregateStruct();

        CoalescedBlockMatrixSkel skel(et.spanStart, et.lumpToSpan, et.colStart,
                                      et.rowParam);

        uint64_t totData = skel.chainData[skel.chainData.size() - 1];
        vector<double> data(totData);

        mt19937 gen(39 + i);
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
        skel.damp(data, 0, sortedSs.ptrs.size() * 2);

        Eigen::MatrixXd verifyMat = skel.densify(data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

        uint64_t largestIndep = findLargestIndependentLumpSet(skel, 0).first;
        Solver solver(std::move(skel),
                      std::vector<uint64_t>{0, largestIndep},  //
                      genOps());
        solver.factor(data.data());
        Eigen::MatrixXd computedMat = solver.skel.densify(data);

        ASSERT_NEAR(
            Eigen::MatrixXd(
                (verifyMat - computedMat).triangularView<Eigen::Lower>())
                .norm(),
            0, 1e-5);
    }
}

TEST(Factor, SparseElimAndFactor_Many_Blas) {
    testSparseElimAndFactor_Many([] { return blasOps(); });
}

TEST(Factor, SparseElimAndFactor_Many_Ref) {
    testSparseElimAndFactor_Many([] { return simpleOps(); });
}