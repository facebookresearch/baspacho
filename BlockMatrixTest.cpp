
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "BlockMatrix.h"


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
