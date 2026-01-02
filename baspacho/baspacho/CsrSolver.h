/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>
#include "baspacho/baspacho/CsrTypes.h"
#include "baspacho/baspacho/Solver.h"

namespace BaSpaCho {

/**
 * Block CSR matrix descriptor - holds structure metadata (no numeric data).
 * Modeled after cuDSS cudssMatrixCreateCsr parameters.
 *
 * This describes a block-sparse matrix in CSR format where:
 * - rowStart[i] gives the index in colIndices where row i's blocks begin
 * - colIndices[rowStart[i]..rowStart[i+1]) are the column indices of blocks in row i
 * - blockSizes[i] gives the dimension of the i-th parameter block
 *
 * The total matrix dimension is sum(blockSizes).
 */
struct BlockCsrDescriptor {
  int64_t numBlocks;         // Number of block rows/cols (square matrix)
  int64_t numBlockNonzeros;  // Number of non-zero blocks
  const void* rowStart;      // Row start pointers [numBlocks+1], type per indexType
  const void* colIndices;    // Column indices [numBlockNonzeros], type per indexType
  const int64_t* blockSizes; // Size of each block [numBlocks]
  IndexType indexType;       // INT32 or INT64
  MatrixType mtype;          // GENERAL, SYMMETRIC, SPD (only SPD supported)
  MatrixView mview;          // FULL, LOWER, UPPER
  IndexBase indexBase;       // ZERO or ONE based

  BlockCsrDescriptor()
      : numBlocks(0),
        numBlockNonzeros(0),
        rowStart(nullptr),
        colIndices(nullptr),
        blockSizes(nullptr),
        indexType(INDEX_INT64),
        mtype(MTYPE_SPD),
        mview(MVIEW_LOWER),
        indexBase(BASE_ZERO) {}
};

/**
 * Create a solver from block-level CSR structure.
 *
 * This is the primary cuDSS-style interface for block CSR matrices.
 * The descriptor provides only the sparsity structure; numeric values
 * are loaded separately via Solver::loadFromCsr() or the accessor.
 *
 * @param settings       Solver settings (backend, threading, fill policy)
 * @param desc           Block CSR descriptor (structure only)
 * @param sparseElimRanges Optional ranges for sparse elimination optimization
 * @param elimLastIds    Optional IDs to keep at end for partial factorization
 * @return               Unique pointer to solver
 *
 * @throws std::invalid_argument if desc has invalid parameters
 */
SolverPtr createSolverFromBlockCsr(const Settings& settings, const BlockCsrDescriptor& desc,
                                   const std::vector<int64_t>& sparseElimRanges = {},
                                   const std::unordered_set<int64_t>& elimLastIds = {});

/**
 * Create a solver from block-level CSR with values preloaded.
 *
 * Convenience function that creates solver and loads initial values.
 * The values array should contain dense block data in CSR order:
 * - Blocks are in row-major order within each block
 * - Blocks appear in the order specified by rowStart/colIndices
 *
 * @param settings       Solver settings
 * @param desc           Block CSR descriptor (structure only)
 * @param values         Numeric values for all blocks (row-major within each block)
 * @param outData        Output data buffer (will be resized to solver.dataSize())
 * @param sparseElimRanges Optional sparse elimination ranges
 * @return               Unique pointer to solver
 *
 * @throws std::invalid_argument if desc has invalid parameters
 */
template <typename T>
SolverPtr createSolverFromBlockCsrWithValues(const Settings& settings,
                                             const BlockCsrDescriptor& desc, const T* values,
                                             std::vector<T>& outData,
                                             const std::vector<int64_t>& sparseElimRanges = {});

/**
 * Convert block CSR descriptor to SparseStructure.
 *
 * Internal helper function that converts the CSR format to BaSpaCho's
 * internal SparseStructure representation.
 *
 * @param desc Block CSR descriptor
 * @return SparseStructure in lower triangular CSR format
 */
SparseStructure blockCsrToSparseStructure(const BlockCsrDescriptor& desc);

/**
 * Get parameter sizes from block CSR descriptor.
 *
 * @param desc Block CSR descriptor
 * @return Vector of block sizes
 */
std::vector<int64_t> getParamSizes(const BlockCsrDescriptor& desc);

}  // namespace BaSpaCho
