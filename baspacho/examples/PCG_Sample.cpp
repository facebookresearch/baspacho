
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
#include "baspacho/examples/Preconditioner.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

void runTest(const std::string& precondArg, int seed) {
  cout << "creating problem..." << endl;
  int numParams = 215;
  auto colBlocks = randomCols(numParams, 0.03, 57 + seed);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 3, 47);

  cout << "creating solver..." << endl;
  auto solver = createSolver(
      {.backend = BackendRef, .addFillPolicy = AddFillForAutoElims}, paramSize,
      ss, {0, 100});

  int64_t nocross = solver->maxFactorParam();
  cout << "max factor: " << nocross << " / " << paramSize.size() << endl;

  int order = solver->order();
  BASPACHO_CHECK_EQ(solver->factorSkel.spanOffsetInLump[nocross], 0);

  cout << "generating prob..." << endl;
  using T = double;
  vector<T> matData = randomData<T>(solver->dataSize(), -1.0, 1.0, 9 + seed);

  // randomized damping
  {
    auto acc = solver->accessor();
    mt19937 gen(seed);
    uniform_real_distribution<> dis(solver->order() * 0.1,
                                    solver->order() * 0.5);
    for (int64_t i = 0; i < (int64_t)paramSize.size(); i++) {
      acc.plainAcc.diagBlock(matData.data(), i).diagonal().array() += dis(gen);
    }
  }

  // b
  vector<T> bData = randomData<T>(order * 1, -1.0, 1.0, 49 + seed);
  Vector<T> b = Eigen::Map<Vector<T>>(bData.data(), order, 1);
  Vector<T> x = b;

  cout << "solve first part..." << endl;

  // factor and solve up to "nocross" value
  vector<T> origMatData = matData;
  solver->factorUpTo(matData.data(), nocross);
  solver->solveLUpTo(matData.data(), nocross, x.data(), order, 1);

  cout << "setup pcg..." << endl;

  std::unique_ptr<Preconditioner<double>> precond;
  if (precondArg == "none") {
    precond.reset(new IdentityPrecond<double>(*solver, nocross));
  } else if (precondArg == "jacobi") {
    precond.reset(new BlockJacobiPrecond<double>(*solver, nocross));
  } else if (precondArg == "gauss-seidel") {
    precond.reset(new BlockGaussSeidelPrecond<double>(*solver, nocross));
  } else if (precondArg == "lower-prec-solve") {
    precond.reset(new LowerPrecSolvePrecond<double>(*solver, nocross));
  } else {
    std::cout << "no such preconditioner '" << precondArg << "'" << std::endl;
    return;
  }
  precond->init(matData.data());

  // set up PCG in elimination area
  int64_t secStart = solver->paramVecDataStart(nocross);
  int64_t secSize = order - secStart;
  PCG pcg(
      [&](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        u.resize(v.size());
        (*precond)(u.data(), v.data());
      },
      [&](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        u.resize(v.size());
        u.setZero();
        solver->addMvFrom(matData.data(), nocross, v.data() - secStart, order,
                          u.data() - secStart, order, 1);
      },
      1e-10, 40, true);

  cout << "run pcg..." << endl;

  Eigen::VectorXd tmp;
  pcg.solve(tmp, x.segment(secStart, secSize));
  x.segment(secStart, secSize) = tmp;

  cout << "solve Lt..." << endl;

  solver->solveLtUpTo(matData.data(), nocross, x.data(), order, 1);

  cout << "comp residual..." << endl;

  Vector<T> b2(order);
  b2.setZero();
  solver->addMvFrom(origMatData.data(), 0, x.data(), order, b2.data(), order,
                    1);

  cout << "rel residual: " << (b - b2).norm() / b.norm() << endl;
}

int main(int argc, char* argv[]) {
  int i = 1;
  std::string precond = "none";
  while (i < argc) {
    if (!strcmp(argv[i], "-p")) {
      if (i == argc - 1) {
        std::cout << "Missing arg to -p argument (can be one of: "
                     "none,jacobi,gauss-seidel,lower-prec-solve)"
                  << std::endl;
        return 1;
      }
      precond = argv[i + 1];
      i += 2;
    } else if (!strcmp(argv[i], "-h")) {
      std::cout
          << "Arguments:\n"
          << "  -p preconditioner (none,jacobi,gauss-seidel,lower-prec-solve)\n"
          << std::endl;
      return 0;
    } else {
      std::cout << "Unknown arg " << argv[i] << " (help: -h)" << std::endl;
    }
  }
  runTest(precond, 37);
  return 0;
}