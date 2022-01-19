#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <limits>
#include <vector>

constexpr uint64_t kInvalid = std::numeric_limits<uint64_t>::max();

template <typename T>
using MatRMaj =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
    Notation for block matrix with grouped columns:
    Linear data:
    * a `span` is the basic grouping of data (at creation)
    * an `range` is formed by a few consecutive params
    Block data:
    * a `block` is `span rows` x `span cols`
    * a `slice` is `span rows` x `range cols`
    * a `slab` is (non-empty) and formed by:
      `all the spans belonging to an range of rows` x `range cols`

    Note that numeric data in a column of slices are a row-major
    matrix. In this way we can refer to the slice sub-matrix,
    or to the whole set of columns as the slice data are consecutive.
*/
struct BlockMatrixSkel {
    BlockMatrixSkel(const std::vector<uint64_t>& spanStart,
                    const std::vector<uint64_t>& rangeToSpan,
                    const std::vector<uint64_t>& colPtr,
                    const std::vector<uint64_t>& rowInd);

    Eigen::MatrixXd densify(const std::vector<double>& data);

    void damp(std::vector<double>& data, double alpha, double beta);

    std::vector<uint64_t> spanStart;  // (with final el)
    std::vector<uint64_t> spanToRange;
    std::vector<uint64_t> rangeStart;   // (with final el)
    std::vector<uint64_t> rangeToSpan;  // (with final el)

    // per-slice data, column-ordered
    std::vector<uint64_t> sliceColPtr;       // slab col data start (with end)
    std::vector<uint64_t> sliceRowSpan;      // row-span id
    std::vector<uint64_t> sliceData;         // numeric data offset
    std::vector<uint64_t> sliceRowsTillEnd;  // num of rows till end

    // per-slab data, column-ordered, colums have a final element
    std::vector<uint64_t> slabColPtr;       // slab col data start (with end)
    std::vector<uint64_t> slabRowRange;     // row-range id (end = invalid)
    std::vector<uint64_t> slabSliceColOrd;  // slice ord in col (end = #slices)

    // per-slab data, row-ordered
    std::vector<uint64_t> slabRowPtr;    // slab row data start (with end)
    std::vector<uint64_t> slabColRange;  // slab's col range
    std::vector<uint64_t> slabColOrd;    // slab order in col
};

BlockMatrixSkel initBlockMatrixSkel(const std::vector<uint64_t>& spanStart,
                                    const std::vector<uint64_t>& rangeToSpan,
                                    const std::vector<uint64_t>& colPtr,
                                    const std::vector<uint64_t>& rowInd);

Eigen::MatrixXd densify(const BlockMatrixSkel& skel,
                        const std::vector<double>& data);

void damp(const BlockMatrixSkel& skel, std::vector<double>& data, double alpha,
          double beta);
