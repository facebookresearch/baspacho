
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <numeric>
#include <sstream>
#include "BlockMatrix.h"
#include "BlockStructure.h"
#include <Eigen/Eigenvalues>


#if 0
//CHEAT SHEET:

//checks that vector v is {5, 10, 15}
ASSERT_THAT(v, ElementsAre(5, 10, 15));

//checks that map m only have elements 1 => 10, 2 => 20
ASSERT_THAT(m, ElementsAre(Pair(1, 10), Pair(2, 20)));

//checks that in vector v all the elements are greater than 10 and less than 20
ASSERT_THAT(v, Each(AllOf(Gt(10), Lt(20))));

//checks that vector v consist of 
//   5, number greater than 10, anything.
ASSERT_THAT(v, ElementsAre(5, Gt(10), _));

ASSERT_THAT(std::vector<uint8_t>(buffer, buffer + buffer_size), 
            ::testing::ElementsAreArray(expect));

//comparison
EXPECT_THAT(foo, testing::UnorderedElementsAreArray(result, 3));
//other way
EXPECT_THAT(foo, testing::ContainerEq(result));
#endif


using namespace std;
using namespace testing;


TEST(BlockMatrix, BasicAssertions) {
    // sizes:                    1    1  2    1    2  2    3    2   2
    vector<uint64_t> paramStart {0,   1, 2,   4,   5, 7,   9,   12, 14, 16};
    // sizes:                          1  3  1  4  3  4
    vector<uint64_t> aggregParamStart {0, 1, 3, 4, 6, 7, 9};
    vector<vector<uint64_t>> columnParams {
        {0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8}, {4, 5, 7}, {6, 8}, {7, 8} };
    BlockMatrixSkel skel = initBlockMatrixSkel(paramStart, aggregParamStart, columnParams);

    ASSERT_THAT(skel.paramStart, ContainerEq(paramStart));
    ASSERT_THAT(skel.aggregParamStart, ContainerEq(aggregParamStart));
    ASSERT_THAT(skel.paramToAggreg, ElementsAre(0, 1, 1, 2, 3, 3, 4, 5, 5));
    ASSERT_THAT(skel.aggregStart, ElementsAre(0, 1, 4, 5, 9, 12, 16));
    
    ASSERT_THAT(skel.blockColDataPtr, ElementsAre(0, 5, 10, 14, 17, 19, 21));
    ASSERT_THAT(skel.blockRowParam, ElementsAre(0, 1, 2, 5, 8, 1, 2, 3, 6, 7, 3, 4, 5, 8, 4, 5, 7, 6, 8, 7, 8));
    // 1x{1, 1, 2, 2, 2}, 3x{1, 2, 1, 3, 2}, 1x{1, 2, 2, 2}, 4x{2, 2, 2}, 3x{3, 2}, 4x{2, 2}
    ASSERT_THAT(skel.blockData, ElementsAre(0, 1, 2, 4, 6, 8, 11, 17, 20, 29, 35, 36, 38, 40, 42, 50, 58, 66, 75, 81, 89, 97));
    
    ASSERT_THAT(skel.blockColGatheredDataPtr, ElementsAre(0, 4, 8, 11, 13, 15, 16));
    ASSERT_THAT(skel.blockRowAggreg, ElementsAre(0, 1, 3, 5,   1, 2, 4, 5,   2, 3, 5,   3, 5,   4, 5,   5));
    ASSERT_THAT(skel.blockRowAggregParamPtr, ElementsAre(0, 1, 3, 4,  0, 2, 3, 4,   0, 1, 3,   0, 2,   0, 1,   0));
}


TEST(BlockMatrix, Densify) {
    vector<uint64_t> paramStart {0,   1, 2,   4,   5, 7,   9,   12, 14, 16};
    vector<uint64_t> aggregParamStart {0, 1, 3, 4, 6, 7, 9};
    vector<vector<uint64_t>> columnParams {
        {0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8}, {4, 5, 7}, {6, 8}, {7, 8} };
    BlockMatrixSkel skel = initBlockMatrixSkel(paramStart, aggregParamStart, columnParams);

    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);
    Eigen::MatrixXd mat = densify(skel, data);

    std::stringstream ss;
    ss << mat;
    std::string expected = 
        " 13   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n"
        " 14  21  22  23   0   0   0   0   0   0   0   0   0   0   0   0\n"
        " 15  24  25  26   0   0   0   0   0   0   0   0   0   0   0   0\n"
        " 16  27  28  29   0   0   0   0   0   0   0   0   0   0   0   0\n"
        "  0  30  31  32  48   0   0   0   0   0   0   0   0   0   0   0\n"
        "  0   0   0   0  49  55  56  57  58   0   0   0   0   0   0   0\n"
        "  0   0   0   0  50  59  60  61  62   0   0   0   0   0   0   0\n"
        " 17   0   0   0  51  63  64  65  66   0   0   0   0   0   0   0\n"
        " 18   0   0   0  52  67  68  69  70   0   0   0   0   0   0   0\n"
        "  0  33  34  35   0   0   0   0   0  79  80  81   0   0   0   0\n"
        "  0  36  37  38   0   0   0   0   0  82  83  84   0   0   0   0\n"
        "  0  39  40  41   0   0   0   0   0  85  86  87   0   0   0   0\n"
        "  0  42  43  44   0  71  72  73  74   0   0   0  94  95  96  97\n"
        "  0  45  46  47   0  75  76  77  78   0   0   0  98  99 100 101\n"
        " 19   0   0   0  53   0   0   0   0  88  89  90 102 103 104 105\n"
        " 20   0   0   0  54   0   0   0   0  91  92  93 106 107 108 109";

    ASSERT_EQ(ss.str(), expected);
}


TEST(BlockMatrix, Damp) {
    vector<uint64_t> paramStart {0,   1, 2,   4,   5, 7,   9,   12, 14, 16};
    vector<uint64_t> aggregParamStart {0, 1, 3, 4, 6, 7, 9};
    vector<vector<uint64_t>> columnParams {
        {0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8}, {4, 5, 7}, {6, 8}, {7, 8} };
    BlockMatrixSkel skel = initBlockMatrixSkel(paramStart, aggregParamStart, columnParams);

    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    Eigen::MatrixXd mat = densify(skel, data);

    double alpha = 2.0, beta = 100.0;
    damp(skel, data, alpha, beta);
    
    Eigen::MatrixXd matDamped = densify(skel, data);

    mat.diagonal() *= (1.0 + alpha);
    mat.diagonal().array() += beta;
    ASSERT_NEAR((mat - matDamped).norm(), 0, 1e-5);
}


TEST(BlockMatrix, SymbolicCholeskyFillIn) {
    vector<uint64_t> paramSize {1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 2};
    vector<set<uint64_t>> colBlocks {
        {0, 6, 7}, {1, 8, 11}, {2, 7, 10}, {3, 11}, {4, 7, 10}, {5, 9}, {6, 7}, {7}, {8}, {9}, {10}, {11}
    };
    BlockStructure blockStruct(paramSize, colBlocks);

    vector<uint64_t> aggregParamStart {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Eigen::MatrixXd verifyMat, computeMat;
    {
        GroupedBlockStructure gbs(blockStruct, aggregParamStart);
        BlockMatrixSkel skel = initBlockMatrixSkel(gbs.paramStart, gbs.aggregParamStart, gbs.columnParams);
        uint64_t totData = skel.blockData[skel.blockData.size() - 1];
        vector<double> data(totData, 1);
        
        std::cout << "original:\n" << densify(skel, data) << std::endl;

        // for comparison compute (dense) cholesky with eigen, and put 1 where we have a nonzero
        damp(skel, data, 0, 1000);
        verifyMat = densify(skel, data);
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(verifyMat);
        for(size_t i = 0; i < verifyMat.size(); i++) {
            verifyMat.data()[i] = (verifyMat.data()[i] != 0);
        }
    }
    
    std::cout << "verification:\n" << verifyMat << std::endl;
    
    {
        blockStruct.addBlocksForEliminationOfRange(0, paramSize.size());
        GroupedBlockStructure gbs(blockStruct, aggregParamStart);
        BlockMatrixSkel skel = initBlockMatrixSkel(gbs.paramStart, gbs.aggregParamStart, gbs.columnParams);
        uint64_t totData = skel.blockData[skel.blockData.size() - 1];
        vector<double> data(totData, 1);
        computeMat = densify(skel, data);
    }
    std::cout << "computed:\n" << computeMat << std::endl;
    
    ASSERT_NEAR((verifyMat - computeMat).norm(), 0, 1e-5);
}

TEST(BlockMatrix, SymbolicCholeskyFillInGrouped) {
    vector<uint64_t> paramSize {1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 2};
    vector<set<uint64_t>> colBlocks {
        {0, 6, 7}, {1, 8, 11}, {2, 7, 10}, {3, 11}, {4, 7, 10}, {5, 9}, {6, 7}, {7}, {8}, {9}, {10}, {11}
    };
    BlockStructure blockStruct(paramSize, colBlocks);

    vector<uint64_t> aggregParamStart {0, 1, 2, 3, 4, 5, 6, 8, 10, 12};
    blockStruct.addBlocksForEliminationOfRange(0, paramSize.size());
    GroupedBlockStructure gbs(blockStruct, aggregParamStart);
    BlockMatrixSkel skel = initBlockMatrixSkel(gbs.paramStart, gbs.aggregParamStart, gbs.columnParams);
    uint64_t totData = skel.blockData[skel.blockData.size() - 1];
    vector<double> data(totData, 1);
    
    std::stringstream ss;
    ss << densify(skel, data);
    std::string expected =
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0\n"
        "1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0\n"
        "1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0\n"
        "1 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0\n"
        "1 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0\n"
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n"
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n"
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n"
        "0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0\n"
        "0 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1\n"
        "0 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1\n"
        "0 0 0 1 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1\n"
        "0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1\n"
        "0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1";

    std::cout << "computed:\n" << ss.str() << std::endl;    
    ASSERT_EQ(ss.str(), expected);
}
