/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <sstream>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/testing/TestingUtils.h"

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

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using namespace ::testing;

TEST(CoalescedBlockMatrix, BasicAssertions) {
  // sizes:                     1  1  2  1  2  2  3   2   2
  vector<int64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
  // sizes:                          1  3  1  4  3  4
  vector<int64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
  vector<set<int64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8},
                                    {4, 5, 7},       {6, 8},          {7, 8}};
  SparseStructure sStruct = columnsToCscStruct(columnParams);
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

  ASSERT_THAT(skel.spanStart, ContainerEq(spanStart));
  ASSERT_THAT(skel.lumpToSpan, ContainerEq(lumpToSpan));
  ASSERT_THAT(skel.spanToLump, ElementsAre(0, 1, 1, 2, 3, 3, 4, 5, 5, 6));
  ASSERT_THAT(skel.lumpStart, ElementsAre(0, 1, 4, 5, 9, 12, 16));

  ASSERT_THAT(skel.chainColPtr, ElementsAre(0, 5, 10, 14, 17, 19, 21));
  ASSERT_THAT(skel.chainRowSpan, ElementsAre(0, 1, 2, 5, 8,  //
                                             1, 2, 3, 6, 7,  //
                                             3, 4, 5, 8,     //
                                             4, 5, 7,        //
                                             6, 8,           //
                                             7, 8));
  // 1x{1, 1, 2, 2, 2}, 3x{1, 2, 1, 3, 2}, 1x{1, 2, 2, 2}, 4x{2, 2, 2}, 3x{3,
  // 2}, 4x{2, 2}
  ASSERT_THAT(skel.chainData, ElementsAre(0, 1, 2, 4, 6, 8, 11, 17, 20, 29, 35, 36, 38, 40, 42, 50,
                                          58, 66, 75, 81, 89, 97));
  ASSERT_THAT(skel.chainRowsTillEnd,
              ElementsAre(1, 2, 4, 6, 8, 1, 3, 4, 7, 9, 1, 3, 5, 7, 2, 4, 6, 3, 5, 2, 4));

  /*
      X
      X X
      _ X X
      X _ X X
      _ X _ _ X
      X X X X X X
  */
  ASSERT_THAT(skel.boardColPtr, ElementsAre(0, 5, 10, 14, 17, 20, 22));
  ASSERT_THAT(skel.boardRowLump, ElementsAre(0, 1, 3, 5, kInvalid,  //
                                             1, 2, 4, 5, kInvalid,  //
                                             2, 3, 5, kInvalid,     //
                                             3, 5, kInvalid,        //
                                             4, 5, kInvalid,        //
                                             5, kInvalid));
  ASSERT_THAT(skel.boardChainColOrd, ElementsAre(0, 1, 3, 4, 5,  //
                                                 0, 2, 3, 4, 5,  //
                                                 0, 1, 3, 4,     //
                                                 0, 2, 3,        //
                                                 0, 1, 2,        //
                                                 0, 2));

  ASSERT_THAT(skel.boardRowPtr, ElementsAre(0, 1, 3, 5, 8, 10, 16));
  ASSERT_THAT(skel.boardColLump, ElementsAre(0,        //
                                             0, 1,     //
                                             1, 2,     //
                                             0, 2, 3,  //
                                             1, 4,     //
                                             0, 1, 2, 3, 4, 5));
  ASSERT_THAT(skel.boardColOrd, ElementsAre(0,        //
                                            1, 0,     //
                                            1, 0,     //
                                            2, 1, 0,  //
                                            2, 0,     //
                                            3, 3, 2, 1, 1, 0));
}

TEST(CoalescedBlockMatrix, Densify) {
  vector<int64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
  vector<int64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
  vector<set<int64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8},
                                    {4, 5, 7},       {6, 8},          {7, 8}};
  SparseStructure sStruct = columnsToCscStruct(columnParams);
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

  int64_t totData = skel.chainData[skel.chainData.size() - 1];
  vector<double> data(totData);
  iota(data.begin(), data.end(), 13);
  Eigen::MatrixXd mat = skel.densify(data);

  Eigen::MatrixXd expectedMat(16, 16);
  expectedMat << 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   //
      14, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           //
      15, 24, 25, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           //
      16, 27, 28, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           //
      0, 30, 31, 32, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           //
      0, 0, 0, 0, 49, 55, 56, 57, 58, 0, 0, 0, 0, 0, 0, 0,          //
      0, 0, 0, 0, 50, 59, 60, 61, 62, 0, 0, 0, 0, 0, 0, 0,          //
      17, 0, 0, 0, 51, 63, 64, 65, 66, 0, 0, 0, 0, 0, 0, 0,         //
      18, 0, 0, 0, 52, 67, 68, 69, 70, 0, 0, 0, 0, 0, 0, 0,         //
      0, 33, 34, 35, 0, 0, 0, 0, 0, 79, 80, 81, 0, 0, 0, 0,         //
      0, 36, 37, 38, 0, 0, 0, 0, 0, 82, 83, 84, 0, 0, 0, 0,         //
      0, 39, 40, 41, 0, 0, 0, 0, 0, 85, 86, 87, 0, 0, 0, 0,         //
      0, 42, 43, 44, 0, 71, 72, 73, 74, 0, 0, 0, 94, 95, 96, 97,    //
      0, 45, 46, 47, 0, 75, 76, 77, 78, 0, 0, 0, 98, 99, 100, 101,  //
      19, 0, 0, 0, 53, 0, 0, 0, 0, 88, 89, 90, 102, 103, 104, 105,  //
      20, 0, 0, 0, 54, 0, 0, 0, 0, 91, 92, 93, 106, 107, 108, 109;

  ASSERT_NEAR((mat - expectedMat).norm(), 0, 1e-10);
}

TEST(CoalescedBlockMatrix, Densify2) {
  vector<int64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
  vector<int64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
  vector<set<int64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8},
                                    {4, 5, 7},       {6, 8},          {7, 8}};
  SparseStructure sStruct = columnsToCscStruct(columnParams);
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

  int64_t totData = skel.chainData[skel.chainData.size() - 1];
  vector<double> data(totData);
  iota(data.begin(), data.end(), 13);
  Eigen::MatrixXd mat;
  skel.densify(mat, data.data(), true, 1);

  Eigen::MatrixXd expectedMat(15, 15);
  expectedMat << 21, 24, 27, 30, 0, 0, 0, 0, 33, 36, 39, 42, 45, 0, 0,  //
      24, 25, 28, 31, 0, 0, 0, 0, 34, 37, 40, 43, 46, 0, 0,             //
      27, 28, 29, 32, 0, 0, 0, 0, 35, 38, 41, 44, 47, 0, 0,             //
      30, 31, 32, 48, 49, 50, 51, 52, 0, 0, 0, 0, 0, 53, 54,            //
      0, 0, 0, 49, 55, 59, 63, 67, 0, 0, 0, 71, 75, 0, 0,               //
      0, 0, 0, 50, 59, 60, 64, 68, 0, 0, 0, 72, 76, 0, 0,               //
      0, 0, 0, 51, 63, 64, 65, 69, 0, 0, 0, 73, 77, 0, 0,               //
      0, 0, 0, 52, 67, 68, 69, 70, 0, 0, 0, 74, 78, 0, 0,               //
      33, 34, 35, 0, 0, 0, 0, 0, 79, 82, 85, 0, 0, 88, 91,              //
      36, 37, 38, 0, 0, 0, 0, 0, 82, 83, 86, 0, 0, 89, 92,              //
      39, 40, 41, 0, 0, 0, 0, 0, 85, 86, 87, 0, 0, 90, 93,              //
      42, 43, 44, 0, 71, 72, 73, 74, 0, 0, 0, 94, 98, 102, 106,         //
      45, 46, 47, 0, 75, 76, 77, 78, 0, 0, 0, 98, 99, 103, 107,         //
      0, 0, 0, 53, 0, 0, 0, 0, 88, 89, 90, 102, 103, 104, 108,          //
      0, 0, 0, 54, 0, 0, 0, 0, 91, 92, 93, 106, 107, 108, 109;

  ASSERT_NEAR((mat - expectedMat).norm(), 0, 1e-10);
}

TEST(CoalescedBlockMatrix, Damp) {
  vector<int64_t> spanStart{0, 1, 2, 4, 5, 7, 9, 12, 14, 16};
  vector<int64_t> lumpToSpan{0, 1, 3, 4, 6, 7, 9};
  vector<set<int64_t>> columnParams{{0, 1, 2, 5, 8}, {1, 2, 3, 6, 7}, {3, 4, 5, 8},
                                    {4, 5, 7},       {6, 8},          {7, 8}};
  SparseStructure sStruct = columnsToCscStruct(columnParams);
  CoalescedBlockMatrixSkel skel(spanStart, lumpToSpan, sStruct.ptrs, sStruct.inds);

  int64_t totData = skel.chainData[skel.chainData.size() - 1];
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
