/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <limits>
#include <vector>
#include "baspacho/baspacho/Accessor.h"

namespace BaSpaCho {

constexpr int64_t kInvalid = -1;

template <typename T>
using MatRMaj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
    Notation for (symmetric) block matrix with coalesced columns:
    Linear data:
    * a `span` is the basic grouping of data (at creation)
    * an `lump` is formed by a few consecutive params
    Block data:
    * a `block` is `span rows` x `span cols`
    * a `chain` is `span rows` x `lump cols`
    * a `board` is (non-empty) and formed by:
      `all the spans belonging to an lump of rows` x `lump cols`

    Note that numeric data in a column of chains are a row-major
    matrix. In this way we can refer to the chain sub-matrix,
    or to the whole set of columns as the chain data are consecutive.
*/
struct CoalescedBlockMatrixSkel {
  CoalescedBlockMatrixSkel(const std::vector<int64_t>& spanStart,
                           const std::vector<int64_t>& lumpToSpan,
                           const std::vector<int64_t>& colPtr, const std::vector<int64_t>& rowInd);

  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> densify(const std::vector<T>& data) const;

  template <typename T>
  void damp(std::vector<T>& data, T alpha, T beta) const;

  // return number of `spans` (ie basic parameter blocks)
  int64_t numSpans() const { return spanStart.size() - 1; }

  // return number of `lumps` (ie aggregated parameter blocks)
  int64_t numLumps() const { return lumpStart.size() - 1; }

  // order of the matrix (sum of all parameter sizes)
  int64_t order() const { return spanStart.back(); }

  // storage of all matrix numeric data
  int64_t dataSize() const { return chainData.back(); }

  // vector storage start for given parameter
  int64_t spanVectorOffset(int64_t span) const { return spanStart[span]; }

  // matrix storage start for given parameter
  int64_t spanMatrixOffset(int64_t span) const {
    int64_t lump = spanToLump[span];
    BASPACHO_CHECK_EQ(spanOffsetInLump[span], 0);
    return chainData[chainColPtr[lump]];
  }

  // return lightweight accessor
  CoalescedAccessor accessor() const {
    CoalescedAccessor retv;
    retv.init(spanStart.data(), spanToLump.data(), lumpStart.data(), spanOffsetInLump.data(),
              chainColPtr.data(), chainRowSpan.data(), chainData.data());
    return retv;
  }

  std::vector<int64_t> spanStart;         // (with final el)
  std::vector<int64_t> spanToLump;        // (with final el)
  std::vector<int64_t> lumpStart;         // (with final el)
  std::vector<int64_t> lumpToSpan;        // (with final el)
  std::vector<int64_t> spanOffsetInLump;  // (with final el)

  // per-chain data, column-ordered
  std::vector<int64_t> chainColPtr;       // board col data start (with end)
  std::vector<int64_t> chainRowSpan;      // row-span id
  std::vector<int64_t> chainData;         // numeric data offset
  std::vector<int64_t> chainRowsTillEnd;  // num of rows till end

  // per-board data, column-ordered, colums have a final element
  std::vector<int64_t> boardColPtr;       // board col data start (with end)
  std::vector<int64_t> boardRowLump;      // row-lump id (end = invalid)
  std::vector<int64_t> boardChainColOrd;  // chain ord in col (end = #chains)

  // per-board data, row-ordered
  std::vector<int64_t> boardRowPtr;   // board row data start (with end)
  std::vector<int64_t> boardColLump;  // board's col lump
  std::vector<int64_t> boardColOrd;   // board order in col
};

}  // end namespace BaSpaCho