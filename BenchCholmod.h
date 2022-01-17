
#include "SparseStructure.h"

std::pair<double, double> benchmarkCholmodSolve(
    const std::vector<uint64_t>& paramSize, const SparseStructure& ss,
    bool verbose = true);