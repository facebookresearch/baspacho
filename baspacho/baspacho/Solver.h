
#include <memory>
#include <unordered_set>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho {

struct Solver {
  Solver(CoalescedBlockMatrixSkel&& factorSkel,
         std::vector<int64_t>&& elimLumpRanges,
         std::vector<int64_t>&& permutation, OpsPtr&& ops);

  PermutedCoalescedAccessor accessor() const {
    PermutedCoalescedAccessor retv;
    retv.init(factorSkel.accessor(), permutation.data());
    return retv;
  }

  PermutedCoalescedAccessor deviceAccessor() const {
    return symCtx->deviceAccessor();
  }

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
  void solveLUpTo(const T* data, int64_t paramIndex, T* vecData, int64_t stride,
                  int nRHS) const;

  template <typename T>
  void solveLtUpTo(const T* data, int64_t paramIndex, T* vecData,
                   int64_t stride, int nRHS) const;

  template <typename T>
  void addMvFrom(const T* matData, int64_t paramIndex, const T* inVecData,
                 int64_t inStride, T* outVecData, int64_t outStride, int nRHS,
                 BaseType<T> alpha = 1.0) const;

  int64_t order() const { return factorSkel.order(); }

  int64_t dataSize() const { return factorSkel.dataSize(); }

 private:
  void initElimination();

  int64_t boardElimTempSize(int64_t lump, int64_t boardIndexInSN) const;

  template <typename T>
  void factorLump(NumericCtx<T>& numCtx, T* data, int64_t lump) const;

  template <typename T>
  void eliminateBoard(NumericCtx<T>& numCtx, T* data, int64_t ptr) const;

  template <typename T>
  void internalSolveLUpTo(SolveCtx<T>& slvCtx, const T* data,
                          int64_t paramIndex, T* vecData, int64_t stride) const;

  template <typename T>
  void internalSolveLtUpTo(SolveCtx<T>& slvCtx, const T* data,
                           int64_t paramIndex, T* vecData,
                           int64_t stride) const;

 public:
  CoalescedBlockMatrixSkel factorSkel;
  std::vector<int64_t> elimLumpRanges;
  std::vector<int64_t> permutation;  // *on indices*: v'[p[i]] = v[i];

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

struct Settings {
  bool findSparseEliminationRanges = true;
  int numThreads = 16;
  BackendType backend = BackendBlas;
};

SolverPtr createSolver(const Settings& settings,
                       const std::vector<int64_t>& paramSize,
                       const SparseStructure& ss);

SolverPtr createSolverSchur(
    const Settings& settings, const std::vector<int64_t>& paramSize,
    const SparseStructure& ss, const std::vector<int64_t>& elimLumpRanges,
    const std::unordered_set<int64_t>& elimLastIds = {});

}  // end namespace BaSpaCho