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

TEST(Solver, Solver) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> paramStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> aggregParamStart{0, 2, 4, 6};
    SparseStructure groupedSs = columnsToCscStruct(
        joinColums(csrStructToColumns(ss), aggregParamStart));
    BlockMatrixSkel skel(paramStart, aggregParamStart, groupedSs.ptrs,
                         groupedSs.inds);

    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    skel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    Solver solver(std::move(skel), std::vector<uint64_t>{}, simpleOps());
    solver.factor(data.data());

    Eigen::MatrixXd computedMat = solver.skel.densify(data);
    std::cout << "COMPUT:\n" << computedMat << std::endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Solver, SolverXt) {
    auto colBlocks = randomCols(100, 0.15, 37);
    SparseStructure ss = columnsToCscStruct(colBlocks).transpose();

    vector<uint64_t> permutation = ss.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedSs = ss.symmetricPermutation(invPerm, false);

    vector<uint64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 5, 47);
    EliminationTree et(paramSize, sortedSs);
    et.buildTree();
    et.computeMerges();
    et.computeAggregateStruct();

    BlockMatrixSkel skel(et.paramStart, et.aggregParamStart, et.colStart,
                         et.rowParam);

    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);

    mt19937 gen(39);
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }
    skel.damp(data, 0, sortedSs.ptrs.size() * 2);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
    std::cout << "VERIF:\n" << verifyMat << std::endl;

    Solver solver(std::move(skel), std::vector<uint64_t>{}, simpleOps());
    solver.factor(data.data());

    Eigen::MatrixXd computedMat = solver.skel.densify(data);
    std::cout << "COMPUT:\n" << computedMat << std::endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}
