#pragma once

#include <memory>
#include <unordered_set>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct Solver {
  Solver(CoalescedBlockMatrixSkel&& factorSkel, std::vector<int64_t>&& elimLumpRanges,
         std::vector<int64_t>&& permutation, int64_t canFactorUpTo, OpsPtr&& ops);

  Solver(CoalescedBlockMatrixSkel&& factorSkel, std::vector<int64_t>&& elimLumpRanges,
         std::vector<int64_t>&& permutation, OpsPtr&& ops)
      : Solver(std::move(factorSkel), std::move(elimLumpRanges), std::move(permutation), -1,
               std::move(ops)) {}

  PermutedCoalescedAccessor accessor() const {
    PermutedCoalescedAccessor retv;
    retv.init(factorSkel.accessor(), permutation.data());
    return retv;
  }

  PermutedCoalescedAccessor deviceAccessor() const { return symCtx->deviceAccessor(); }

  void printStats() const;

  void resetStats();

  template <typename T>
  void factor(T* data, bool verbose = false) const;

  template <typename T>
  void solve(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void solveL(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void solveLt(const T* matData, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void factorUpTo(T* data, int64_t paramIndex, bool verbose = false) const;

  template <typename T>
  void solveLUpTo(const T* data, int64_t paramIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void solveLtUpTo(const T* data, int64_t paramIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void addMvFrom(const T* matData, int64_t paramIndex, const T* inVecData, int64_t inStride,
                 T* outVecData, int64_t outStride, int nRHS, BaseType<T> alpha = 1.0) const;

  template <typename T>
  void pseudoFactorFrom(T* data, int64_t paramIndex, bool verbose = false) const;

  template <typename T>
  void factorFrom(T* data, int64_t paramIndex, bool verbose = false) const;

  template <typename T>
  void solveLFrom(const T* data, int64_t paramIndex, T* vecData, int64_t stride, int nRHS) const;

  template <typename T>
  void solveLtFrom(const T* data, int64_t paramIndex, T* vecData, int64_t stride, int nRHS) const;

  int64_t order() const { return factorSkel.order(); }

  int64_t dataSize() const { return factorSkel.dataSize(); }

  int64_t maxFactorParam() const { return canFactorUpTo; }

  int64_t paramVecDataStart(int64_t paramIndex) const {
    return factorSkel.paramVecDataStart(paramIndex);
  }

  int64_t paramMatDataStart(int64_t paramIndex) const {
    return factorSkel.paramMatDataStart(paramIndex);
  }

  CoalescedBlockMatrixSkel& skel() { return factorSkel; }

  const CoalescedBlockMatrixSkel& skel() const { return factorSkel; }

  const std::vector<int64_t>& sparseEliminationLumpRanges() const { return elimLumpRanges; }

  const std::vector<int64_t>& paramPermutation() const { return permutation; }

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
  void internalFactorRange(T* data, int64_t startParamIndex, int64_t endParamIndex,
                           bool verbose = false) const;

  template <typename T>
  void internalSolveLRange(SolveCtx<T>& slvCtx, const T* data, int64_t startParamIndex,
                           int64_t endParamIndex, T* vecData, int64_t stride) const;

  template <typename T>
  void internalSolveLtRange(SolveCtx<T>& slvCtx, const T* data, int64_t startParamIndex,
                            int64_t endParamIndex, T* vecData, int64_t stride) const;

  CoalescedBlockMatrixSkel factorSkel;
  std::vector<int64_t> elimLumpRanges;
  std::vector<int64_t> permutation;  // *on indices*: v'[p[i]] = v[i];
  int64_t canFactorUpTo;

  OpsPtr ops;
  SymbolicCtxPtr symCtx;
  std::vector<SymElimCtxPtr> elimCtxs;
  std::vector<int64_t> startElimRowPtr;
  int64_t maxElimTempSize;
};

using SolverPtr = std::unique_ptr<Solver>;

enum BackendType {
  BackendRef,
  BackendBlas,
  BackendCuda,
};

enum AddFillPolicy {
  AddFillComplete,       // add fill for complete factoring, reorder
  AddFillForAutoElims,   // add fill for give+auto elim-ranges, reorder
  AddFillForGivenElims,  // fill for elimination of elim ranges, no reorder
  AddFillNone,           // no fill added, no reorder
};

// if not set, a default will be selected depending on the backend
struct ComputationModel;

struct Settings {
  bool findSparseEliminationRanges = true;
  int numThreads = 16;
  BackendType backend = BackendBlas;
  AddFillPolicy addFillPolicy = AddFillComplete;
  ComputationModel* computationModel = nullptr;
};

SolverPtr createSolver(const Settings& settings, const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss, const std::vector<int64_t>& elimLumpRanges = {},
                       const std::unordered_set<int64_t>& elimLastIds = {});

}  // end namespace BaSpaCho