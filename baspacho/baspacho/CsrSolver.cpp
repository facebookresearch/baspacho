/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/CsrSolver.h"
#include <stdexcept>
#include <string>

namespace BaSpaCho {

namespace {

// Helper to convert indices from void* based on IndexType
template <typename OutT>
void copyIndices(std::vector<OutT>& out, const void* src, int64_t count, IndexType indexType,
                 int64_t baseAdjust) {
  out.resize(count);
  if (indexType == INDEX_INT32) {
    const int32_t* srcTyped = static_cast<const int32_t*>(src);
    for (int64_t i = 0; i < count; i++) {
      out[i] = static_cast<OutT>(srcTyped[i]) - baseAdjust;
    }
  } else {
    const int64_t* srcTyped = static_cast<const int64_t*>(src);
    for (int64_t i = 0; i < count; i++) {
      out[i] = static_cast<OutT>(srcTyped[i]) - baseAdjust;
    }
  }
}

// Validate the block CSR descriptor
void validateDescriptor(const BlockCsrDescriptor& desc) {
  // Check for null pointers
  if (desc.numBlocks > 0 && desc.rowStart == nullptr) {
    throw std::invalid_argument("BlockCsrDescriptor: rowStart is null");
  }
  if (desc.numBlockNonzeros > 0 && desc.colIndices == nullptr) {
    throw std::invalid_argument("BlockCsrDescriptor: colIndices is null");
  }
  if (desc.numBlocks > 0 && desc.blockSizes == nullptr) {
    throw std::invalid_argument("BlockCsrDescriptor: blockSizes is null");
  }

  // Check non-negative counts
  if (desc.numBlocks < 0) {
    throw std::invalid_argument("BlockCsrDescriptor: numBlocks must be non-negative");
  }
  if (desc.numBlockNonzeros < 0) {
    throw std::invalid_argument("BlockCsrDescriptor: numBlockNonzeros must be non-negative");
  }

  // Validate matrix type/view combination
  validateMatrixTypeView(desc.mtype, desc.mview);

  // Check block sizes are positive
  for (int64_t i = 0; i < desc.numBlocks; i++) {
    if (desc.blockSizes[i] <= 0) {
      throw std::invalid_argument("BlockCsrDescriptor: blockSizes[" + std::to_string(i) +
                                  "] must be positive, got " + std::to_string(desc.blockSizes[i]));
    }
  }
}

}  // namespace

SparseStructure blockCsrToSparseStructure(const BlockCsrDescriptor& desc) {
  validateDescriptor(desc);

  if (desc.numBlocks == 0) {
    return SparseStructure({0}, {});
  }

  // Determine base adjustment for 1-based indexing
  int64_t baseAdjust = (desc.indexBase == BASE_ONE) ? 1 : 0;

  // Copy row pointers (ptrs)
  std::vector<int64_t> ptrs;
  copyIndices(ptrs, desc.rowStart, desc.numBlocks + 1, desc.indexType, baseAdjust);

  // Validate row pointers
  if (ptrs[0] != 0) {
    throw std::invalid_argument("BlockCsrDescriptor: rowStart[0] must be 0 (after base adjustment)");
  }
  if (ptrs[desc.numBlocks] != desc.numBlockNonzeros) {
    throw std::invalid_argument(
        "BlockCsrDescriptor: rowStart[numBlocks] must equal numBlockNonzeros");
  }

  // Copy column indices (inds)
  std::vector<int64_t> inds;
  copyIndices(inds, desc.colIndices, desc.numBlockNonzeros, desc.indexType, baseAdjust);

  // Validate column indices
  for (int64_t i = 0; i < desc.numBlockNonzeros; i++) {
    if (inds[i] < 0 || inds[i] >= desc.numBlocks) {
      throw std::invalid_argument("BlockCsrDescriptor: colIndices[" + std::to_string(i) +
                                  "] out of range");
    }
  }

  // Create SparseStructure (currently in CSR format)
  SparseStructure ss(std::move(ptrs), std::move(inds));

  // Handle matrix view:
  // - MVIEW_LOWER: Already in correct format for BaSpaCho (lower triangular CSR)
  // - MVIEW_UPPER: Need to transpose to get lower triangular
  // - MVIEW_FULL: For SPD, we only need lower triangle, so clear upper
  if (desc.mview == MVIEW_UPPER) {
    // Transpose converts upper CSR to lower CSR
    ss = ss.transpose();
  } else if (desc.mview == MVIEW_FULL) {
    // For full matrix, extract lower triangle only
    // clear(false) clears upper half, keeping lower
    ss = ss.clear(false);
  }

  return ss;
}

std::vector<int64_t> getParamSizes(const BlockCsrDescriptor& desc) {
  if (desc.numBlocks == 0) {
    return {};
  }
  return std::vector<int64_t>(desc.blockSizes, desc.blockSizes + desc.numBlocks);
}

SolverPtr createSolverFromBlockCsr(const Settings& settings, const BlockCsrDescriptor& desc,
                                   const std::vector<int64_t>& sparseElimRanges,
                                   const std::unordered_set<int64_t>& elimLastIds) {
  // Convert block CSR to SparseStructure
  SparseStructure ss = blockCsrToSparseStructure(desc);

  // Get parameter sizes
  std::vector<int64_t> paramSizes = getParamSizes(desc);

  // Delegate to existing createSolver
  return createSolver(settings, paramSizes, ss, sparseElimRanges, elimLastIds);
}

// Template instantiations for createSolverFromBlockCsrWithValues
template <typename T>
SolverPtr createSolverFromBlockCsrWithValues(const Settings& settings,
                                             const BlockCsrDescriptor& desc, const T* values,
                                             std::vector<T>& outData,
                                             const std::vector<int64_t>& sparseElimRanges) {
  // Create solver (structure only)
  SolverPtr solver = createSolverFromBlockCsr(settings, desc, sparseElimRanges, {});

  // Resize output data buffer
  outData.resize(solver->dataSize());

  // Zero-initialize (fill-in entries need to be zero)
  std::fill(outData.begin(), outData.end(), T(0));

  // Convert row pointers to int64_t for loadFromCsr
  int64_t baseAdjust = (desc.indexBase == BASE_ONE) ? 1 : 0;
  std::vector<int64_t> rowStart(desc.numBlocks + 1);
  std::vector<int64_t> colIndices(desc.numBlockNonzeros);

  if (desc.indexType == INDEX_INT32) {
    const int32_t* rowStartSrc = static_cast<const int32_t*>(desc.rowStart);
    const int32_t* colIndicesSrc = static_cast<const int32_t*>(desc.colIndices);
    for (int64_t i = 0; i <= desc.numBlocks; i++) {
      rowStart[i] = static_cast<int64_t>(rowStartSrc[i]) - baseAdjust;
    }
    for (int64_t i = 0; i < desc.numBlockNonzeros; i++) {
      colIndices[i] = static_cast<int64_t>(colIndicesSrc[i]) - baseAdjust;
    }
  } else {
    const int64_t* rowStartSrc = static_cast<const int64_t*>(desc.rowStart);
    const int64_t* colIndicesSrc = static_cast<const int64_t*>(desc.colIndices);
    for (int64_t i = 0; i <= desc.numBlocks; i++) {
      rowStart[i] = rowStartSrc[i] - baseAdjust;
    }
    for (int64_t i = 0; i < desc.numBlockNonzeros; i++) {
      colIndices[i] = colIndicesSrc[i] - baseAdjust;
    }
  }

  // Load values from CSR format into internal format
  solver->loadFromCsr(rowStart.data(), colIndices.data(), desc.blockSizes, values, outData.data());

  return solver;
}

// Explicit template instantiations
template SolverPtr createSolverFromBlockCsrWithValues<float>(const Settings& settings,
                                                             const BlockCsrDescriptor& desc,
                                                             const float* values,
                                                             std::vector<float>& outData,
                                                             const std::vector<int64_t>& sparseElimRanges);

template SolverPtr createSolverFromBlockCsrWithValues<double>(const Settings& settings,
                                                              const BlockCsrDescriptor& desc,
                                                              const double* values,
                                                              std::vector<double>& outData,
                                                              const std::vector<int64_t>& sparseElimRanges);

}  // namespace BaSpaCho
