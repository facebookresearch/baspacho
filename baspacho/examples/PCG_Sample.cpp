
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/examples/PCG.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

static constexpr int minNumSparseElimNodes = 50;

void runTest(int seed) {
  int numParams = 215;
  auto colBlocks = randomCols(numParams, 0.03, 57 + seed);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
  SparseStructure sortedSs = columnsToCscStruct(colBlocks).transpose();

  cout << "generating prob..." << endl;

  // test no-cross barrier - make sure the elim set is still present
  mt19937 gen(seed);
  uniform_int_distribution<> dis(minNumSparseElimNodes + 5, 210);
  int64_t nocross = dis(gen);

  vector<int64_t> paramSize = randomVec(sortedSs.ptrs.size() - 1, 2, 3, 47);
  EliminationTree et(paramSize, sortedSs);
  et.buildTree();
  et.computeMerges(/* compute sparse elim ranges = */ true, {nocross});
  et.computeAggregateStruct();

  CoalescedBlockMatrixSkel factorSkel(et.computeSpanStart(), et.lumpToSpan,
                                      et.colStart, et.rowParam);
  int order = factorSkel.order();
  BASPACHO_CHECK_EQ(factorSkel.spanOffsetInLump[nocross], 0);

  using T = double;
  vector<T> data = randomData<T>(factorSkel.dataSize(), -1.0, 1.0, 9 + seed);
  factorSkel.damp(data, T(0.0), T(factorSkel.order() * 0.1));
  Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, blasOps());

  // b
  vector<T> bData = randomData<T>(order * 1, -1.0, 1.0, 49 + seed);
  Vector<T> b = Eigen::Map<Vector<T>>(bData.data(), order, 1);
  Vector<T> x = b;

  cout << "solve first part..." << endl;

  // factor and solve up to "nocross" value
  vector<T> origData = data;
  solver.factorUpTo(data.data(), nocross);
  solver.solveLUpTo(data.data(), nocross, x.data(), order, 1);

  cout << "setup pcg..." << endl;

  // set up PCG in elimination area
  int64_t secStart = solver.paramVecDataStart(nocross);
  int64_t secSize = order - secStart;
  PCG pcg([&](Eigen::VectorXd& u, const Eigen::VectorXd& v) { u = v; },
          [&](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
            u.resize(v.size());
            u.setZero();
            solver.addMvFrom(data.data(), nocross, v.data() - secStart, order,
                             u.data() - secStart, order, 1);
          },
          1e-10, 40, true);

  cout << "run pcg..." << endl;

  Eigen::VectorXd tmp;
  pcg.solve(tmp, x.segment(secStart, secSize));
  x.segment(secStart, secSize) = tmp;

  cout << "solve Lt..." << endl;

  solver.solveLtUpTo(data.data(), nocross, x.data(), order, 1);

  cout << "comp residual..." << endl;

  Vector<T> b2(order);
  b2.setZero();
  solver.addMvFrom(origData.data(), 0, x.data(), order, b2.data(), order, 1);

  cout << "rel residual: " << (b - b2).norm() / b.norm() << endl;

  /*Matrix<T> mat = factorSkel.densify(data);
  int order = factorSkel.order();
  int barrierAt = factorSkel.spanStart[nocross];
  int afterBar = order - barrierAt;

  ASSERT_GE(et.sparseElimRanges.size(), 2);
  int64_t largestIndep = et.sparseElimRanges[1];
  Solver solver(move(factorSkel), move(et.sparseElimRanges), {}, genOps());*/
}

int main(int argc, char* argv[]) {
  runTest(37);
  return 0;
}