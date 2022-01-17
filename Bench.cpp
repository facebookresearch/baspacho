
#include <iomanip>
#include <iostream>

#include "BenchCholmod.h"
#include "TestingUtils.h"

using namespace std;

// todo: try with different generator with different sparse topologies
void runBenchmark(int numRuns, uint64_t size, uint64_t paramSize, double fill) {
    vector<double> analysisCholmodTimings, factorCholmodTimings;
    cout << "Benchmark (size=" << size << ", pSize: " << paramSize
         << ", fill: " << fill << ");" << endl;
    for (int i = 0; i < numRuns; i++) {
        cout << "\r" << setfill('.') << setw(i) << ""
             << "(" << i << "/" << numRuns << ")" << flush;

        int seed = i + 37;
        auto columns = randomCols(size, fill, seed);
        SparseStructure ss = columnsToCscStruct(columns).transpose();

        vector<uint64_t> paramSizes(size, paramSize);
        auto timings = benchmarkCholmodSolve(paramSizes, ss, false);
        analysisCholmodTimings.push_back(timings.first);
        factorCholmodTimings.push_back(timings.second);
    }
    cout << "\r" << setfill('.') << setw(numRuns) << ""
         << "(" << numRuns << "/" << numRuns << ")" << endl;

    cout << "analysis, Cholmod:\n" << printVec(analysisCholmodTimings) << endl;
    cout << "factor, Cholmod:\n" << printVec(factorCholmodTimings) << endl;
}

int main(int argc, char* argv[]) {
    runBenchmark(30, 500, 5, 0.1);

    return 0;
}