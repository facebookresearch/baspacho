#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <limits>
#include <vector>

constexpr uint64_t kInvalid = std::numeric_limits<uint64_t>::max();

template <typename T>
using MatRMaj =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct BlockMatrixSkel {
    BlockMatrixSkel(const std::vector<uint64_t>& paramStart,
                    const std::vector<uint64_t>& aggregParamStart,
                    const std::vector<uint64_t>& colPtr,
                    const std::vector<uint64_t>& rowInd);

    Eigen::MatrixXd densify(const std::vector<double>& data);

    void damp(std::vector<double>& data, double alpha, double beta);

    std::vector<uint64_t> paramStart;        // num_params + 1
    std::vector<uint64_t> paramToAggreg;     // num_params
    std::vector<uint64_t> aggregStart;       // num_aggregs + 1
    std::vector<uint64_t> aggregParamStart;  // num_aggregs + 1

    // A matrix block is identified by a pair of param x aggreg
    std::vector<uint64_t> blockColDataPtr;       // num_aggregs + 1
    std::vector<uint64_t> blockRowParam;         // num_blocks
    std::vector<uint64_t> blockData;             // num_blocks + 1
    std::vector<uint64_t> endBlockNumRowsAbove;  // num_blocks

    // We also need to know about the "gathered" blocks, where we have
    // grouped the consecutive row params into aggregates.
    // This is because we will process the colum of blocks taking not
    // one row at a time, but an aggregate of rows.
    std::vector<uint64_t> blockColGatheredDataPtr;  // num_aggregs + 1
    std::vector<uint64_t> blockRowAggreg;  // num_gathered_blocks + num_aggregs
    std::vector<uint64_t>
        blockRowAggregParamPtr;  // num_gathered_blocks + num_aggregs
};

BlockMatrixSkel initBlockMatrixSkel(
    const std::vector<uint64_t>& paramStart,
    const std::vector<uint64_t>& aggregParamStart,
    const std::vector<uint64_t>& colPtr, const std::vector<uint64_t>& rowInd);

Eigen::MatrixXd densify(const BlockMatrixSkel& skel,
                        const std::vector<double>& data);

void damp(const BlockMatrixSkel& skel, std::vector<double>& data, double alpha,
          double beta);
