#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <sstream>

#include "BlockMatrix.h"
#include "Factor.h"
#include "SparseStructure.h"
#include "TestingUtils.h"

using namespace std;

TEST(Factor, FactorAggreg) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> paramStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> aggregParamStart{0, 2, 4, 6};
    SparseStructure groupedSs = columnsToCscStruct(
        joinColums(csrStructToColumns(ss), aggregParamStart));
    BlockMatrixSkel skel = initBlockMatrixSkel(paramStart, aggregParamStart,
                                               groupedSs.ptrs, groupedSs.inds);
    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);
    damp(skel, data, 5, 50);

    Eigen::MatrixXd verifyMat = densify(skel, data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    factorAggreg(skel, data, 0);
    Eigen::MatrixXd computedMat = densify(skel, data);
    std::cout << computedMat << std::endl;

    ASSERT_NEAR((verifyMat - computedMat).leftCols(5).norm(), 0, 1e-5);
}

TEST(Factor, Factor) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> paramStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> aggregParamStart{0, 2, 4, 6};
    SparseStructure groupedSs = columnsToCscStruct(
        joinColums(csrStructToColumns(ss), aggregParamStart));
    BlockMatrixSkel skel = initBlockMatrixSkel(paramStart, aggregParamStart,
                                               groupedSs.ptrs, groupedSs.inds);

    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);
    std::cout << "VERIF:\n" << densify(skel, data) << std::endl;

    damp(skel, data, 5, 50);

    Eigen::MatrixXd verifyMat = densify(skel, data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
    std::cout << "VERIF:\n" << verifyMat << std::endl;

    factor(skel, data);
    Eigen::MatrixXd computedMat = densify(skel, data);
    std::cout << "COMPUT:\n" << computedMat << std::endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}
