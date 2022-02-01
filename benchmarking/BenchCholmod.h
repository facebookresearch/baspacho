
#include "../baspacho/SparseStructure.h"

std::pair<double, double> benchmarkCholmodSolve(
    const std::vector<int64_t>& paramSize, const SparseStructure& ss,
    bool verbose = true);