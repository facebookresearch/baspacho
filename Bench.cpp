
#include <glog/logging.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

#include "BenchCholmod.h"
#include "Solver.h"
#include "TestingMatGen.h"
#include "TestingUtils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct SparseProblem {
    SparseStructure sparseStruct;
    vector<uint64_t> paramSize;
};

pair<double, double> benchmarkSolver(const SparseProblem& prob, bool verbose) {
    auto startAnalysis = hrc::now();
    SolverPtr solver = createSolver(prob.paramSize, prob.sparseStruct, verbose);
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // generate mock data
    uint64_t totData =
        solver->skel.chainData[solver->skel.chainData.size() - 1];
    vector<double> data = randomData(totData, -1.0, 1.0, 37);
    uint64_t order = solver->skel.spanStart[solver->skel.spanStart.size() - 1];
    solver->skel.damp(data, 0, order * 1.2);  // make positive def

    auto startFactor = hrc::now();
    solver->factor(data.data(), verbose);
    double factorTime = tdelta(hrc::now() - startFactor).count();

    if (verbose) {
        solver->ops->printStats();
        LOG_IF(INFO, verbose) << "analysis: " << analysisTime
                              << ", factor: " << factorTime << endl;
    }

    return std::make_pair(analysisTime, factorTime);
}

SparseProblem matGenToSparseProblem(SparseMatGenerator& gen, uint64_t pSizeMin,
                                    uint64_t pSizeMax) {
    SparseProblem retv;
    retv.sparseStruct = columnsToCscStruct(gen.columns).transpose();
    if (pSizeMin == pSizeMax) {
        retv.paramSize.assign(gen.columns.size(), pSizeMin);
    } else {
        retv.paramSize =
            randomVec(gen.columns.size(), pSizeMin, pSizeMax, gen.gen);
    }
    return retv;
}

map<string, function<SparseProblem(int64_t)>> problemGenerators = {
    {"flat_size=1000_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
};

map<string, function<pair<double, double>(const SparseProblem&, bool)>>
    solvers = {
        {"BaSpaCho_BLAS_nth=16", benchmarkSolver},  //
        {"CHOLMOD",
         [](const SparseProblem& prob, bool verbose) -> pair<double, double> {
             return benchmarkCholmodSolve(prob.paramSize, prob.sparseStruct,
                                          verbose);
         }},  //
};

void runBenchmarks(int numIterations, bool verbose = true, int seed = 37) {
    for (auto [probName, gen] : problemGenerators) {
        map<string, vector<double>> factorTimings;
        int prevLen = 0;
        for (int it = 0; it < numIterations; it++) {
            SparseProblem prob = gen(seed + it * 1000000);
            int solvIdx = 1;
            for (auto [solvName, solv] : solvers) {
                if (verbose) {
                    cout << endl;
                }

                stringstream ss;
                ss << setfill('.') << setw(it + 1) << ""
                   << "(" << it + 1 << "/" << numIterations << ", " << solvName
                   << " " << solvIdx << "/" << solvers.size() << ")";
                solvIdx++;
                string str = ss.str();
                int clearSize = max(0, prevLen - (int)str.size());
                prevLen = str.size();
                cout << "\r" << str << setfill(' ') << setw(clearSize) << ""
                     << flush;
                if (verbose) {
                    cout << endl;
                }

                auto [analysisT, factorT] = solv(prob, verbose);
                factorTimings[solvName].push_back(factorT);
            }
        }
        stringstream ss;
        ss << setfill('.') << setw(numIterations) << ""
           << "(" << numIterations << "/" << numIterations << ", done!)";
        string str = ss.str();
        int clearSize = max(0, prevLen - (int)str.size());
        cout << "\r" << str << setfill(' ') << setw(clearSize) << "" << endl;

        for (auto [solvName, timings] : factorTimings) {
            cout << solvName << ":\n  " << printVec(timings) << endl;
        }
    }
}

#if 0
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
#endif

int main(int argc, char* argv[]) {
    runBenchmarks(10);
    // runBenchmark(10, 1000, 3, 1);
    // runBenchmark(1, 10000, 2, 0.05, 7500);
    // runBenchmark(10, 2000, 3, 1);

    return 0;
}