
#include <glog/logging.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>

#include "../baspacho/Solver.h"
#include "../testing/TestingMatGen.h"
#include "../testing/TestingUtils.h"
#include "BenchCholmod.h"

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

    // random entries
    {"01_flat_size=1000_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"02_flat_size=4000_fill=0.01_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genFlat(4000, 0.01, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"03a_flat_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
         gen.addSchurSet(50000, 0.02);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"03b_flat_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.0002",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
         gen.addSchurSet(5000, 0.0002);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //

    // base is grid
    {"04_grid_size=100x100_fill=1.0_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen =
             SparseMatGenerator::genGrid(100, 100, 1.0, 2, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"05_grid_size=150x150_fill=1.0_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen =
             SparseMatGenerator::genGrid(150, 150, 1.0, 2, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"06_grid_size=200x200_fill=0.25_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen =
             SparseMatGenerator::genGrid(200, 200, 0.25, 2, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"07_grid_size=200x200_fill=0.05_conn=3_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen =
             SparseMatGenerator::genGrid(150, 150, 0.05, 3, seed);
         return matGenToSparseProblem(gen, 3, 3);
     }},  //

    // base is meridians
    {"08_meri_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
         SparseMatGenerator gen = SparseMatGenerator::genMeridians(
             4, 1500, 0.5, 120, 600, 2, 2, seed);
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

struct BenchmarkSettings {
    int numIterations = 5;
    int verbose = true;
    regex selectProblems = regex("");
    regex excludeProblems;
    regex selectSolvers = regex("");
    regex excludeSolvers;
    string referenceSolver;
};

void runBenchmarks(const BenchmarkSettings& settings, int seed = 37) {
    if (!settings.referenceSolver.empty()) {
        CHECK(solvers.find(settings.referenceSolver) != solvers.end())
            << "Solver '" << settings.referenceSolver
            << "' does not exist or not available at compile time";
    }
    int numSolversToTest = 0;
    for (auto [solvName, solv] : solvers) {
        if (solvName != settings.referenceSolver &&
            (!regex_search(solvName, settings.selectSolvers) ||
             regex_search(solvName, settings.excludeSolvers))) {
            continue;
        }
        numSolversToTest++;
    }

    for (auto [probName, gen] : problemGenerators) {
        if (!regex_search(probName, settings.selectProblems) ||
            regex_search(probName, settings.excludeProblems)) {
            continue;
        }

        cout << "prob: " << probName << endl;

        map<string, vector<double>> factorTimings;
        int prevLen = 0;
        for (int it = 0; it < settings.numIterations; it++) {
            SparseProblem prob = gen(seed + it * 1000000);
            int solvIdx = 1;
            for (auto [solvName, solv] : solvers) {
                if (solvName != settings.referenceSolver &&
                    (!regex_search(solvName, settings.selectSolvers) ||
                     regex_search(solvName, settings.excludeSolvers))) {
                    continue;
                }

                if (settings.verbose) {
                    cout << endl;
                }

                stringstream ss;
                ss << setfill('.') << setw(it + 1) << ""
                   << "(" << it + 1 << "/" << settings.numIterations << ", "
                   << solvName << " " << solvIdx << "/" << numSolversToTest
                   << ")";
                solvIdx++;
                string str = ss.str();
                int clearSize = max(0, prevLen - (int)str.size());
                prevLen = str.size();
                cout << "\r" << str << setfill(' ') << setw(clearSize) << ""
                     << flush;
                if (settings.verbose) {
                    cout << endl;
                }

                auto [analysisT, factorT] = solv(prob, settings.verbose);
                factorTimings[solvName].push_back(factorT);
            }
        }
        stringstream ss;
        ss << setfill('.') << setw(settings.numIterations) << ""
           << "(" << settings.numIterations << "/" << settings.numIterations
           << ", done!)";
        string str = ss.str();
        int clearSize = max(0, prevLen - (int)str.size());
        cout << "\r" << str << setfill(' ') << setw(clearSize) << "" << endl;

        if (settings.referenceSolver.empty()) {
            for (auto [solvName, timings] : factorTimings) {
                cout << solvName << ":\n  " << printVec(timings) << endl;
            }
        } else {
            auto it = factorTimings.find(settings.referenceSolver);
            CHECK(it != factorTimings.end());
            const vector<double>& refTimings = it->second;
            for (auto [solvName, timings] : factorTimings) {
                if (solvName == settings.referenceSolver) {
                    continue;
                }
                vector<double> relTimings(timings.size());
                for (size_t i = 0; i < timings.size(); i++) {
                    relTimings[i] = timings[i] / refTimings[i];
                }
                cout << solvName << " VS " << settings.referenceSolver
                     << ":\n  " << printVec(relTimings) << endl;
            }
        }
    }
}

#if 1
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
        SparseProblem prob;
        prob.paramSize = paramSizes;
        prob.sparseStruct = ss;
        auto timings = benchmarkSolver(prob, false);
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

void list() {
    cout << "Problem generators:" << endl;
    for (auto [probName, gen] : problemGenerators) {
        cout << "  " << probName << endl;
    }
    cout << "Solvers:" << endl;
    for (auto [solvName, solv] : solvers) {
        cout << "  " << solvName << endl;
    }
}

int main(int argc, char* argv[]) {
    BenchmarkSettings settings;
    // settings.referenceSolver = "CHOLMOD";
    settings.verbose = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-l")) {
            list();
            return 0;
        }
        if (!strcmp(argv[i], "-v")) {
            settings.verbose = true;
        } else if (!strcmp(argv[i], "-n") && i < argc - 1) {
            settings.numIterations = stoi(argv[++i]);
        } else if (!strcmp(argv[i], "-R") && i < argc - 1) {
            settings.selectProblems = regex(argv[++i]);
        } else if (!strcmp(argv[i], "-X") && i < argc - 1) {
            settings.excludeProblems = regex(argv[++i]);
        } else if (!strcmp(argv[i], "-S") && i < argc - 1) {
            settings.selectSolvers = regex(argv[++i]);
        } else if (!strcmp(argv[i], "-E") && i < argc - 1) {
            settings.excludeSolvers = regex(argv[++i]);
        }
    }

    runBenchmarks(settings);
    // runBenchmark(10, 1000, 3, 1);
    // runBenchmark(1, 10000, 2, 0.05, 7500);
    // runBenchmark(10, 2000, 3, 1);

    return 0;
}