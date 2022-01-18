
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "BenchCholmod.h"
#include "Solver.h"
#include "TestingUtils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

std::pair<double, double> benchmarkSolver(
    const std::vector<uint64_t>& paramSize, const SparseStructure& ss,
    bool verbose = true) {
    auto startAnalysis = hrc::now();
    SolverPtr solver = createSolver(paramSize, ss);
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // mock data
    uint64_t totData =
        solver->skel.blockData[solver->skel.blockData.size() - 1];
    vector<double> data(totData);

    mt19937 gen(39);
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }
    uint64_t order =
        solver->skel.paramStart[solver->skel.paramStart.size() - 1];
    solver->skel.damp(data, 0, order * 1.2);

    auto startFactor = hrc::now();
    solver->factor(data.data());
    double factorTime = tdelta(hrc::now() - startFactor).count();

    return std::make_pair(analysisTime, factorTime);
}

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

    vector<double> analysisSolverTimings, factorSolverTimings;
    for (int i = 0; i < numRuns; i++) {
        cout << "\r" << setfill('.') << setw(i) << ""
             << "(" << i << "/" << numRuns << ")" << flush;

        int seed = i + 37;
        auto columns = randomCols(size, fill, seed);
        SparseStructure ss = columnsToCscStruct(columns).transpose();

        vector<uint64_t> paramSizes(size, paramSize);
        auto timings = benchmarkSolver(paramSizes, ss, false);
        analysisSolverTimings.push_back(timings.first);
        factorSolverTimings.push_back(timings.second);
    }
    cout << "\r" << setfill('.') << setw(numRuns) << ""
         << "(" << numRuns << "/" << numRuns << ")" << endl;

    cout << "analysis, Solver:\n" << printVec(analysisSolverTimings) << endl;
    cout << "factor, Solver:\n" << printVec(factorSolverTimings) << endl;
}

int main(int argc, char* argv[]) {
    runBenchmark(10, 200, 3, 0.1);

    return 0;
}