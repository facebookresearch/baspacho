
#include <map>

#include "baspacho/baspacho/SparseStructure.h"

struct CholmodBenchResults {
  double analysisTime;
  double factorTime;
  std::map<int64_t, double> solveTimes;
  int nRHS;
};

CholmodBenchResults benchmarkCholmodSolve(
    const std::vector<int64_t>& paramSize, const BaSpaCho::SparseStructure& ss,
    const std::vector<int64_t>& nRHSs = {}, int verbose = 0);