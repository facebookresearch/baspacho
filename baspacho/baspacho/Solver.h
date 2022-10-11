/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_set>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

/**
 * @brief Class solver represents a symbolic decomposition with the operations required to
 * operate on (externally allocated) numeric matrix/vector data.
 *
 * You will never have to create a Solver class yourself, use the createSolver function below,
 * which performs the symbolic analysis in order to create a proper factor, computing a param
 * reordering, and adding required fill to the reordered sparse structure.
 *
 * Note that this is a low-level interface, and does not provide reordering of numeric data,
 * and all solve functions assume internal ordering. This might not be required when solving
 * eg. see optimizer example.
 *
 * The provided parameters are called `spans` in the factor, so for convenience and clarity
 * we say that the reordering maps the user parameter index to the internal span index:
 *   spanIndex = reord[spanIndex].
 */
class Solver {
 public:
  // constructor, from RAW factor skeleton (do not call directly, use createSolver)
  Solver(CoalescedBlockMatrixSkel&& factorSkel, std::vector<int64_t>&& sparseElimRanges,
         std::vector<int64_t>&& permutation, OpsPtr&& ops, int64_t canFactorUpTo = -1);

  // return a (permuted) accessor to access factor's block (re-ordering is auto-applied)
  PermutedCoalescedAccessor accessor() const {
    PermutedCoalescedAccessor retv;
    retv.init(factorSkel.accessor(), permutation.data());
    return retv;
  }

  // return an accessor to be used by an on-device kernel (if supported by backend)
  PermutedCoalescedAccessor deviceAccessor() const { return symCtx->deviceAccessor(); }

  // print some statistics about timings
  void printStats() const;

  // reset statistics
  void resetStats();

  // factor the data stored in the factor
  template <typename T>
  void factor(T* data, bool verbose = false) const;

  // solve in place with LLt (vector must be permuted)
  template <typename T>
  void solve(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // solve in place with L (vector must be permuted)
  template <typename T>
  void solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // solve in place with Lt (vector must be permuted)
  template <typename T>
  void solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  // apply partial factor, up to a given span
  template <typename T>
  void factorUpTo(T* data, int64_t spanIndex, bool verbose = false) const;

  // apply partial solve (lower triangular), up to a given span
  template <typename T>
  void solveLUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // apply partial solve (lower triangular), up to a given span
  template <typename T>
  void solveLtUpTo(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // outVec += M * inVec, applying M's bottom right corner from `spanIndex`
  template <typename T>
  void addMvFrom(const T* matData, int64_t spanIndex, const T* inVecData, int64_t inStride,
                 T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha = 1.0) const;

  // apply pseudo-factor ( /= diagBlockLt where diagBlockLt has Lt factors of diagonal blocks)
  template <typename T>
  void pseudoFactorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex)
  template <typename T>
  void factorFrom(T* data, int64_t spanIndex, bool verbose = false) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex), and
  // vector data from spanVectorOffset(spanIndex)
  template <typename T>
  void solveLFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // factor from given spanIndex, only uses factor data from spanMatrixOffset(spanInedex), and
  // vector data from spanVectorOffset(spanIndex)
  template <typename T>
  void solveLtFrom(const T* data, int64_t spanIndex, T* vecData, int64_t stride, int nRHS) const;

  // order of the factor
  int64_t order() const { return factorSkel.order(); }

  // storge data size
  int64_t dataSize() const { return factorSkel.dataSize(); }

  // returns the upper span index limit for proper factorization (if the factor doesn't have fill
  // for full factorization this might not include all parameters)
  int64_t canFactorUpToSpan() const { return canFactorUpTo; }

  // offset of span vector data
  int64_t spanVectorOffset(int64_t spanIndex) const {
    return factorSkel.spanVectorOffset(spanIndex);
  }

  // offset of span matrix data
  int64_t spanMatrixOffset(int64_t spanIndex) const {
    return factorSkel.spanMatrixOffset(spanIndex);
  }

  // return sparse structure of the factor
  const CoalescedBlockMatrixSkel& skel() const { return factorSkel; }

  // return the (span/lump) ranges set to undergo sparse elimination
  const std::vector<int64_t>& sparseEliminationRanges() const { return sparseElimRanges; }

  // return the reordering applied to parameters (i's position is perm[i] in the factor)
  const std::vector<int64_t>& paramToSpan() const { return permutation; }

  // TESTING: return
  SymbolicCtx& internalSymbolicContext() { return *symCtx; }

  SymElimCtx& internalGetElimCtx(size_t i) {
    BASPACHO_CHECK_LT(i, elimCtxs.size());
    return *elimCtxs[i];
  }

 private:
  void initElimination();

  int64_t boardElimTempSize(int64_t lump, int64_t boardIndexInSN) const;

  template <typename T>
  void factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const;

  template <typename T>
  void eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

  template <typename T>
  void internalFactorRange(T* data, int64_t startSpanIndex, int64_t endSpanIndex,
                           bool verbose = false) const;

  template <typename T>
  void internalSolveLRange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                           int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void internalSolveLtRange(SolveCtx<T>& slvCtx, const T* data, int64_t startSpanIndex,
                            int64_t endSpanIndex, T* vecData, int64_t stride, int nRHS) const;

  CoalescedBlockMatrixSkel factorSkel;
  std::vector<int64_t> sparseElimRanges;
  std::vector<int64_t> permutation;  // *on indices*: v'[p[i]] = v[i];
  int64_t canFactorUpTo;

  OpsPtr ops;
  SymbolicCtxPtr symCtx;
  std::vector<SymElimCtxPtr> elimCtxs;
  std::vector<int64_t> startElimRowPtr;
  int64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

/**
 * The backend type selectes the engine that will be used for numerical operations. Note that
 * device (Cuda) engines will expect memory allocated on the device, and will crash when provided
 * data on the CPU. Ref/Blas engines in the other hand will only work with CPU data.
 **/
enum BackendType {
  BackendRef,  // reference implementation, not recommended
  BackendFast,
  BackendCuda,
};

/**
 * Policy on fill adding to sparse matrix structure. Note that this controls the factor's sparse
 * structure, and therefore if the solver will support total/partial factor
 **/
enum AddFillPolicy {
  AddFillComplete,       // add fill for complete factoring, reorder
  AddFillForAutoElims,   // add fill for give+auto elim-ranges, reorder
  AddFillForGivenElims,  // fill for elimination of elim ranges, no reorder
  AddFillNone,           // no fill added, no reorder
};

// forward def, represent the computation model to tune for
struct ComputationModel;

/**
 * Settings for `createSolver` function below (symbolic analysis of a sparse block matrix).
 */
struct Settings {
  bool findSparseEliminationRanges = true;
  int numThreads = 16;
  BackendType backend = BackendFast;
  AddFillPolicy addFillPolicy = AddFillComplete;
  const ComputationModel* computationModel = nullptr;
};

/**
 * @brief Create a Solver object, performing symbolic analysis, reordering and creating solver
 * with a factor where proper fill has been added to allow full or partial factorization.
 *
 * @param settings settings as explained above
 * @param paramSizes vector with size of the n-th parameter block
 * @param ss a csr structure (ptrs/inds) representing the *blocks*
 * @param sparseElimRanges [a_0, a_1, ..., a_n] where [a_i,a_{i+1}] will be treated as a sparse
 * elimination range
 * @param elimLastIds ids to be kept at the very end in reordering, allowing partial factor
 * that will eliminate all other parameters. Must all be after sparse elimination ranges. If
 * non-empty then the settings.addFillPolicy MUST be AddFillComplete (there is no point in
 * having this set if it's not possible to eliminate up to there).
 * @return SolverPtr
 */
SolverPtr createSolver(const Settings& settings, const std::vector<int64_t>& paramSizes,
                       const SparseStructure& ss, const std::vector<int64_t>& sparseElimRanges = {},
                       const std::unordered_set<int64_t>& elimLastIds = {});

}  // end namespace BaSpaCho
