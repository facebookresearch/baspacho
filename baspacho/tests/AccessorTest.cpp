#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;

using OuterStride = Eigen::OuterStride<>;
using OuterStridedRMajMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;
using OuterStridedCMajMatM = Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    OuterStride>;

TEST(Accessor, CoalescedAccessor) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);
    vector<double> data(skel.dataSize(), 0);

    Eigen::MatrixXd denseMat(skel.order(), skel.order());
    denseMat.setZero();

    int seed = 0;
    auto acc = skel.accessor();
    for (int64_t c = 0; c < colBlocks.size(); c++) {
        int64_t cSize = acc.paramSize(c);
        int64_t cStart = acc.paramStart(c);
        for (int64_t r : colBlocks[c]) {
            int64_t rSize = acc.paramSize(r);
            int64_t rStart = acc.paramStart(r);

            std::vector<double> randomBlockData =
                randomData(rSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    rSize, cSize);

            auto [off, stride] = acc.blockOffset(r, c);
            OuterStridedRMajMatM block(data.data() + off, rSize, cSize,
                                       OuterStride(stride));
            block = randomBlock;
            denseMat.block(rStart, cStart, rSize, cSize) = randomBlock;
        }

        // diagonal
        {
            std::vector<double> randomBlockData =
                randomData(cSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    cSize, cSize);

            auto [off, stride] = acc.blockOffset(c, c);
            OuterStridedRMajMatM block(data.data() + off, cSize, cSize,
                                       OuterStride(stride));
            block += randomBlock;
            denseMat.block(cStart, cStart, cSize, cSize) += randomBlock;
        }
    }

    Eigen::MatrixXd densifiedMat = skel.densify(data);

    ASSERT_NEAR(Eigen::MatrixXd(
                    (denseMat - densifiedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Accessor, PermutedCoalescedAccessor) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);

    vector<int64_t> permutation(colBlocks.size());
    iota(permutation.begin(), permutation.end(), 0);
    mt19937 g(37);
    shuffle(permutation.begin(), permutation.end(), g);
    vector<int64_t> invP = inversePermutation(permutation);

    vector<double> data(skel.dataSize(), 0);
    Eigen::MatrixXd denseMat(skel.order(), skel.order());
    denseMat.setZero();

    int seed = 0;
    PermutedCoalescedAccessor acc;
    acc.init(skel.accessor(), permutation.data());
    for (int64_t pc = 0; pc < colBlocks.size(); pc++) {
        int64_t c = invP[pc];
        int64_t cSize = acc.paramSize(c);
        int64_t cStart = acc.paramStart(c);
        for (int64_t pr : colBlocks[pc]) {
            int64_t r = invP[pr];
            int64_t rSize = acc.paramSize(r);
            int64_t rStart = acc.paramStart(r);
            auto [off, stride, flip] = acc.blockOffset(r, c);
            ASSERT_EQ(flip, pr < pc);

            if (!flip) {
                std::vector<double> randomBlockData =
                    randomData(rSize * cSize, -1.0, 1.0, seed++);
                Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                        rSize, cSize);

                OuterStridedRMajMatM block(data.data() + off, rSize, cSize,
                                           OuterStride(stride));
                block = randomBlock;
                denseMat.block(rStart, cStart, rSize, cSize) = randomBlock;
            } else {
                std::vector<double> randomBlockData =
                    randomData(rSize * cSize, -1.0, 1.0, seed++);
                Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                        cSize, rSize);

                OuterStridedCMajMatM block(data.data() + off, cSize, rSize,
                                           OuterStride(stride));
                block = randomBlock;
                denseMat.block(cStart, rStart, cSize, rSize) = randomBlock;
            }
        }

        // diagonal
        {
            std::vector<double> randomBlockData =
                randomData(cSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    cSize, cSize);

            auto [off, stride, flip] = acc.blockOffset(c, c);
            ASSERT_FALSE(flip);
            OuterStridedRMajMatM block(data.data() + off, cSize, cSize,
                                       OuterStride(stride));
            block += randomBlock;
            denseMat.block(cStart, cStart, cSize, cSize) += randomBlock;
        }
    }

    Eigen::MatrixXd densifiedMat = skel.densify(data);

    ASSERT_NEAR(Eigen::MatrixXd(
                    (denseMat - densifiedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Accessor, BlockCoalescedAccessor) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);
    vector<double> data(skel.dataSize(), 0);

    Eigen::MatrixXd denseMat(skel.order(), skel.order());
    denseMat.setZero();

    auto acc = skel.accessor();

    int seed = 0;
    for (int64_t c = 0; c < colBlocks.size(); c++) {
        int64_t cSize = acc.paramSize(c);
        int64_t cStart = acc.paramStart(c);
        for (int64_t r : colBlocks[c]) {
            int64_t rSize = acc.paramSize(r);
            int64_t rStart = acc.paramStart(r);

            std::vector<double> randomBlockData =
                randomData(rSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    rSize, cSize);

            auto block = acc.block(data.data(), r, c);
            ASSERT_EQ(block.rows(), rSize);
            ASSERT_EQ(block.cols(), cSize);
            block = randomBlock;
            denseMat.block(rStart, cStart, rSize, cSize) = randomBlock;
        }

        // diagonal
        {
            std::vector<double> randomBlockData =
                randomData(cSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    cSize, cSize);

            auto block = acc.diagBlock(data.data(), c);
            ASSERT_EQ(block.rows(), cSize);
            ASSERT_EQ(block.cols(), cSize);
            block += randomBlock;
            denseMat.block(cStart, cStart, cSize, cSize) += randomBlock;
        }
    }

    Eigen::MatrixXd densifiedMat = skel.densify(data);

    ASSERT_NEAR(Eigen::MatrixXd(
                    (denseMat - densifiedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}

TEST(Accessor, BlockPermutedCoalescedAccessor) {
    vector<set<int64_t>> colBlocks{{0, 3, 5}, {1}, {2, 4}, {3}, {4}, {5}};
    SparseStructure ss =
        columnsToCscStruct(colBlocks).transpose().addFullEliminationFill();
    vector<int64_t> spanStart{0, 2, 5, 7, 10, 12, 15};
    vector<int64_t> lumpToSpan{0, 2, 4, 6};
    SparseStructure groupedSs =
        columnsToCscStruct(joinColums(csrStructToColumns(ss), lumpToSpan));
    CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, groupedSs.ptrs,
                                  groupedSs.inds);

    vector<int64_t> permutation(colBlocks.size());
    iota(permutation.begin(), permutation.end(), 0);
    mt19937 g(37);
    shuffle(permutation.begin(), permutation.end(), g);
    vector<int64_t> invP = inversePermutation(permutation);

    vector<double> data(skel.dataSize(), 0);
    Eigen::MatrixXd denseMat(skel.order(), skel.order());
    denseMat.setZero();

    int seed = 0;
    PermutedCoalescedAccessor acc;
    acc.init(skel.accessor(), permutation.data());
    for (int64_t pc = 0; pc < colBlocks.size(); pc++) {
        int64_t c = invP[pc];
        int64_t cSize = acc.paramSize(c);
        int64_t cStart = acc.paramStart(c);
        for (int64_t pr : colBlocks[pc]) {
            int64_t r = invP[pr];
            int64_t rSize = acc.paramSize(r);
            int64_t rStart = acc.paramStart(r);
            std::vector<double> randomBlockData =
                randomData(rSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    rSize, cSize);

            auto block = acc.block(data.data(), r, c);
            ASSERT_EQ(block.rows(), rSize);
            ASSERT_EQ(block.cols(), cSize);
            block = randomBlock;

            bool flip = pr < pc;
            if (!flip) {
                denseMat.block(rStart, cStart, rSize, cSize) = randomBlock;
            } else {
                denseMat.block(cStart, rStart, cSize, rSize) =
                    randomBlock.transpose();
            }
        }

        // diagonal
        {
            std::vector<double> randomBlockData =
                randomData(cSize * cSize, -1.0, 1.0, seed++);
            Eigen::Map<Eigen::MatrixXd> randomBlock(randomBlockData.data(),
                                                    cSize, cSize);

            auto block = acc.diagBlock(data.data(), c);
            ASSERT_EQ(block.rows(), cSize);
            ASSERT_EQ(block.cols(), cSize);
            block += randomBlock;
            denseMat.block(cStart, cStart, cSize, cSize) += randomBlock;
        }
    }

    Eigen::MatrixXd densifiedMat = skel.densify(data);

    ASSERT_NEAR(Eigen::MatrixXd(
                    (denseMat - densifiedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-5);
}