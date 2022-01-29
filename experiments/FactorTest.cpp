#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <sstream>

#include "../baspacho/BlockMatrix.h"
#include "../baspacho/SparseStructure.h"
#include "../testing/TestingUtils.h"
#include "Factor.h"

using namespace std;

TEST(XperFactor, FactorAggreg) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    BlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);
    uint64_t totData = skel.chainData[skel.chainData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);
    skel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);

    factorLump(skel, data, 0);
    Eigen::MatrixXd computedMat = skel.densify(data);
    std::cout << computedMat << std::endl;
    ASSERT_NEAR((verifyMat - computedMat).leftCols(5).norm(), 0, 1e-5);
}

TEST(XperFactor, Factor) {
    vector<set<uint64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<uint64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<uint64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    BlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs, groupedSs.inds);

    uint64_t totData = skel.chainData[skel.chainData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    skel.damp(data, 5, 50);

    Eigen::MatrixXd verifyMat = skel.densify(data);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
    std::cout << "VERIF:\n" << verifyMat << std::endl;

    factor(skel, data);
    Eigen::MatrixXd computedMat = skel.densify(data);
    std::cout << "COMPUT:\n" << computedMat << std::endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}
