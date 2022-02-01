
#include "baspacho/SparseStructure.h"

std::pair<double, double> benchmarkCholmodSolve(
    const std::vector<int64_t>& paramSize, const BaSpaCho::SparseStructure& ss,
    bool verbose = true);