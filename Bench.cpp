
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
    uint64_t schurSize, bool verbose = true) {
    auto startAnalysis = hrc::now();
    SolverPtr solver = createSolver(paramSize, ss);
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // mock data
    uint64_t totData =
        solver->skel.sliceData[solver->skel.sliceData.size() - 1];
    vector<double> data(totData);

    mt19937 gen(39);
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dis(gen);
    }
    uint64_t order = solver->skel.spanStart[solver->skel.spanStart.size() - 1];
    solver->skel.damp(data, 0, order * 1.2);

    auto startFactor = hrc::now();
    solver->factor(data.data());
    double factorTime = tdelta(hrc::now() - startFactor).count();

    cout << endl;
    solver->ops->printStats();
    cout << "analysis: " << analysisTime << ", factor: " << factorTime << endl;

    return std::make_pair(analysisTime, factorTime);
}

// todo: try with different generator with different sparse topologies
void runBenchmark(int numRuns, uint64_t size, uint64_t paramSize, double fill,
                  uint64_t schurSize = 0) {
    vector<double> analysisSolverTimings, factorSolverTimings;
    for (int i = 0; i < numRuns; i++) {
        cout << "\r" << setfill('.') << setw(i) << ""
             << "(" << i << "/" << numRuns << ")" << flush;

        int seed = i + 37;
        auto columns = randomCols(size, fill, seed);
        if (schurSize) {
            columns = makeIndependentElimSet(columns, 0, schurSize);
        }
        SparseStructure ss = columnsToCscStruct(columns).transpose();

        vector<uint64_t> paramSizes(size, paramSize);
        auto timings = benchmarkSolver(paramSizes, ss, schurSize, false);
        analysisSolverTimings.push_back(timings.first);
        factorSolverTimings.push_back(timings.second);
    }
    cout << "\r" << setfill('.') << setw(numRuns) << ""
         << "(" << numRuns << "/" << numRuns << ")" << endl;

    cout << "analysis, Solver:\n" << printVec(analysisSolverTimings) << endl;
    cout << "factor, Solver:\n" << printVec(factorSolverTimings) << endl;

    vector<double> analysisCholmodTimings, factorCholmodTimings;
    cout << "Benchmark (size=" << size << ", pSize: " << paramSize
         << ", fill: " << fill << ");" << endl;
    for (int i = 0; i < numRuns; i++) {
        cout << "\r" << setfill('.') << setw(i) << ""
             << "(" << i << "/" << numRuns << ")" << flush;

        int seed = i + 37;
        auto columns = randomCols(size, fill, seed);
        if (schurSize) {
            columns = makeIndependentElimSet(columns, 0, schurSize);
        }
        SparseStructure ss = columnsToCscStruct(columns).transpose();

        vector<uint64_t> paramSizes(size, paramSize);
        auto timings = benchmarkCholmodSolve(paramSizes, ss, true);
        analysisCholmodTimings.push_back(timings.first);
        factorCholmodTimings.push_back(timings.second);
    }
    cout << "\r" << setfill('.') << setw(numRuns) << ""
         << "(" << numRuns << "/" << numRuns << ")" << endl;

    cout << "analysis, Cholmod:\n" << printVec(analysisCholmodTimings) << endl;
    cout << "factor, Cholmod:\n" << printVec(factorCholmodTimings) << endl;
}

int main(int argc, char* argv[]) {
    // runBenchmark(10, 1000, 3, 1);
    runBenchmark(1, 10000, 2, 0.05, 7500);
    // runBenchmark(10, 2000, 3, 1);

    return 0;
}