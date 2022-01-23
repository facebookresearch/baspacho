#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <sstream>

#include "BlockMatrix.h"
#include "SparseStructure.h"
#include "TestingUtils.h"

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
    // sizes:                     1  1  2  1  2  2  3   2   2
    vector<uint64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
    // sizes:                          1  3  1  4  3  4
    vector<uint64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
    vector<set<uint64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7},
                                       {3, 4, 5, 8},    {4, 5, 7},
                                       {6, 8},          {7, 8}};
    SparseStructure sStruct = columnsToCscStruct(columnParams);
    BlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

    ASSERT_THAT(skel.spanStart, ContainerEq(spanStart));
    ASSERT_THAT(skel.lumpToSpan, ContainerEq(lumpToSpan));
    ASSERT_THAT(skel.spanToLump, ElementsAre(0, 1, 1, 2, 3, 3, 4, 5, 5));
    ASSERT_THAT(skel.lumpStart, ElementsAre(0, 1, 4, 5, 9, 12, 16));

    ASSERT_THAT(skel.sliceColPtr, ElementsAre(0, 5, 10, 14, 17, 19, 21));
    ASSERT_THAT(skel.sliceRowSpan, ElementsAre(0, 1, 2, 5, 8,  //
                                               1, 2, 3, 6, 7,  //
                                               3, 4, 5, 8,     //
                                               4, 5, 7,        //
                                               6, 8,           //
                                               7, 8));
    // 1x{1, 1, 2, 2, 2}, 3x{1, 2, 1, 3, 2}, 1x{1, 2, 2, 2}, 4x{2, 2, 2}, 3x{3,
    // 2}, 4x{2, 2}
    ASSERT_THAT(skel.sliceData,
                ElementsAre(0, 1, 2, 4, 6, 8, 11, 17, 20, 29, 35, 36, 38, 40,
                            42, 50, 58, 66, 75, 81, 89, 97));
    ASSERT_THAT(skel.sliceRowsTillEnd,
                ElementsAre(1, 2, 4, 6, 8, 1, 3, 4, 7, 9, 1, 3, 5, 7, 2, 4, 6,
                            3, 5, 2, 4));

    /*
        X
        X X
        _ X X
        X _ X X
        _ X _ _ X
        X X X X X X
    */
    ASSERT_THAT(skel.slabColPtr, ElementsAre(0, 5, 10, 14, 17, 20, 22));
    ASSERT_THAT(skel.slabRowLump, ElementsAre(0, 1, 3, 5, kInvalid,  //
                                              1, 2, 4, 5, kInvalid,  //
                                              2, 3, 5, kInvalid,     //
                                              3, 5, kInvalid,        //
                                              4, 5, kInvalid,        //
                                              5, kInvalid));
    ASSERT_THAT(skel.slabSliceColOrd, ElementsAre(0, 1, 3, 4, 5,  //
                                                  0, 2, 3, 4, 5,  //
                                                  0, 1, 3, 4,     //
                                                  0, 2, 3,        //
                                                  0, 1, 2,        //
                                                  0, 2));

    ASSERT_THAT(skel.slabRowPtr, ElementsAre(0, 1, 3, 5, 8, 10, 16));
    ASSERT_THAT(skel.slabColLump, ElementsAre(0,        //
                                              0, 1,     //
                                              1, 2,     //
                                              0, 2, 3,  //
                                              1, 4,     //
                                              0, 1, 2, 3, 4, 5));
    ASSERT_THAT(skel.slabColOrd, ElementsAre(0,        //
                                             1, 0,     //
                                             1, 0,     //
                                             2, 1, 0,  //
                                             2, 0,     //
                                             3, 3, 2, 1, 1, 0));
}

TEST(BlockMatrix, Densify) {
    vector<uint64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
    vector<uint64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
    vector<set<uint64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7},
                                       {3, 4, 5, 8},    {4, 5, 7},
                                       {6, 8},          {7, 8}};
    SparseStructure sStruct = columnsToCscStruct(columnParams);
    BlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

    uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);
    Eigen::MatrixXd mat = skel.densify(data);

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
    vector<uint64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
    vector<uint64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
    vector<set<uint64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7},
                                       {3, 4, 5, 8},    {4, 5, 7},
                                       {6, 8},          {7, 8}};
    SparseStructure sStruct = columnsToCscStruct(columnParams);
    BlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

    uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
    vector<double> data(totData);
    iota(data.begin(), data.end(), 13);

    Eigen::MatrixXd mat = skel.densify(data);

    double alpha = 2.0, beta = 100.0;
    skel.damp(data, alpha, beta);

    Eigen::MatrixXd matDamped = skel.densify(data);

    mat.diagonal() *= (1.0 + alpha);
    mat.diagonal().array() += beta;
    ASSERT_NEAR((mat - matDamped).norm(), 0, 1e-5);
}