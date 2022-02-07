
#include "baspacho/SparseStructure.h"

struct CholmodBenchResults {
    double analysisTime;
    double factorTime;
    double solve1Time;
    double solve2Time;
    double solveNRHSTime;
    int nRHS;
};

CholmodBenchResults benchmarkCholmodSolve(const std::vector<int64_t>& paramSize,
                                          const BaSpaCho::SparseStructure& ss,
                                          int verbose = 0);