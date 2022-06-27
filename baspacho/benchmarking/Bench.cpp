
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>

#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/testing/TestingMatGen.h"
#include "baspacho/testing/TestingUtils.h"

#ifdef BASPACHO_USE_CUBLAS
#include "baspacho/baspacho/CudaDefs.h"
#endif

#ifdef BASPACHO_HAVE_CHOLMOD
#include "BenchCholmod.h"
#endif

using namespace BaSpaCho;
using namespace testing;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

struct SparseProblem {
  SparseStructure sparseStruct;
  vector<int64_t> paramSize;
};

struct BenchResults {
  double analysisTime;
  double factorTime;
  map<int64_t, double> solveTimes;
};

// first access to Cuda/Cublas takes a long time
#ifdef BASPACHO_USE_CUBLAS
void bangGpu() {
  static bool doneBang = false;
  if (!doneBang) {
    void* ptr;
    vector<uint8_t> bytes(100000);
    cuCHECK(cudaMalloc(&ptr, bytes.size() * sizeof(uint8_t)));
    cuCHECK(cudaMemcpy(ptr, bytes.data(), bytes.size() * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));
    cuCHECK(cudaMemcpy(bytes.data(), ptr, bytes.size() * sizeof(uint8_t),
                       cudaMemcpyDeviceToHost));
    cuCHECK(cudaFree(ptr));
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverDnH = nullptr;
    cublasCHECK(cublasCreate(&cublasH));
    cusolverCHECK(cusolverDnCreate(&cusolverDnH));
    cublasCHECK(cublasDestroy(cublasH));
    cusolverCHECK(cusolverDnDestroy(cusolverDnH));
    doneBang = true;
  }
}
#endif

BenchResults benchmarkSolver(const SparseProblem& prob,
                             const Settings& settings,
                             const std::vector<int64_t> nRHSs = {},
                             bool verbose = true) {
#ifdef BASPACHO_USE_CUBLAS
  if (settings.backend == BackendCuda) {
    bangGpu();
  }
#endif

  auto startAnalysis = hrc::now();
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.sparseStruct);
  double analysisTime = tdelta(hrc::now() - startAnalysis).count();

  // generate mock data, make positive def
  vector<double> data =
      randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
  solver->factorSkel.damp(data, double(0),
                          double(solver->factorSkel.order() * 1.2));

  double factorTime;
  std::map<int64_t, double> solveTimes;
#ifdef BASPACHO_USE_CUBLAS
  if (settings.backend == BackendCuda) {
    DevMirror dataGpu(data);
    auto startFactor = hrc::now();
    solver->factor(dataGpu.ptr);
    factorTime = tdelta(hrc::now() - startFactor).count();

    for (int64_t nRHS : nRHSs) {
      vector<double> vecData =
          randomData(nRHS * solver->order(), -1.0, 1.0, 38);
      DevMirror vecDataGpu(vecData);

      // heat up
      solver->solve(dataGpu.ptr, vecDataGpu.ptr, solver->order(), nRHS);
      vecDataGpu.load(vecData);

      auto startSolve = hrc::now();
      solver->solve(dataGpu.ptr, vecDataGpu.ptr, solver->order(), nRHS);
      solveTimes[nRHS] = tdelta(hrc::now() - startSolve).count();
    }
  } else
#endif  // BASPACHO_USE_CUBLAS
  {
    auto startFactor = hrc::now();
    solver->factor(data.data(), verbose);
    factorTime = tdelta(hrc::now() - startFactor).count();

    for (int64_t nRHS : nRHSs) {
      vector<double> vecData =
          randomData(nRHS * solver->order(), -1.0, 1.0, 38);

      // heat up
      vector<double> vecDataCp = vecData;
      solver->solve(data.data(), vecDataCp.data(), solver->order(), nRHS);

      auto startSolve = hrc::now();
      solver->solve(data.data(), vecData.data(), solver->order(), nRHS);
      solveTimes[nRHS] = tdelta(hrc::now() - startSolve).count();
    }
  }

  if (verbose) {
    solver->printStats();
    cout << "sparse elim ranges: " << printVec(solver->elimLumpRanges) << endl;
  }

  BenchResults retv;
  retv.analysisTime = analysisTime;
  retv.factorTime = factorTime;
  retv.solveTimes = solveTimes;
  return retv;
}

#ifdef BASPACHO_USE_CUBLAS
BenchResults benchmarkSolverBatched(const SparseProblem& prob,
                                    const Settings& settings, int batchSize,
                                    const std::vector<int64_t> nRHSs = {},
                                    bool verbose = true) {
  bangGpu();

  auto startAnalysis = hrc::now();
  SolverPtr solver = createSolver(settings, prob.paramSize, prob.sparseStruct);
  double analysisTime = tdelta(hrc::now() - startAnalysis).count();

  // generate mock data, make positive def
  int order = solver->order();
  vector<vector<double>> datas(batchSize);
  vector<DevMirror<double>> datasGpu(batchSize);
  vector<double*> datasPtr(batchSize);
  for (int q = 0; q < batchSize; q++) {
    datas[q] =
        randomData<double>(solver->factorSkel.dataSize(), -1.0, 1.0, 37 + q);
    solver->factorSkel.damp(datas[q], double(0), double(order * 1.3));
    datasGpu[q].load(datas[q]);
    datasPtr[q] = datasGpu[q].ptr;
  }

  double factorTime;
  std::map<int64_t, double> solveTimes;

  auto startFactor = hrc::now();
  solver->factor(&datasPtr);
  factorTime = tdelta(hrc::now() - startFactor).count() / batchSize;

  for (int64_t nRHS : nRHSs) {
    vector<vector<double>> rhsDatas(batchSize);
    vector<DevMirror<double>> rhsDatasGpu(batchSize);
    vector<double*> rhsDatasPtr(batchSize);
    for (int q = 0; q < batchSize; q++) {
      rhsDatas[q] = randomData<double>(nRHS * order, -1.0, 1.0, 37 + q + nRHS);
      rhsDatasGpu[q].load(rhsDatas[q]);
      rhsDatasPtr[q] = rhsDatasGpu[q].ptr;
    }

    // heat up
    solver->solve(&datasPtr, &rhsDatasPtr, solver->order(), nRHS);
    for (int q = 0; q < batchSize; q++) {
      rhsDatasGpu[q].load(rhsDatas[q]);
      rhsDatasPtr[q] = rhsDatasGpu[q].ptr;
    }

    auto startSolve = hrc::now();
    solver->solve(&datasPtr, &rhsDatasPtr, solver->order(), nRHS);
    solveTimes[nRHS] = tdelta(hrc::now() - startSolve).count() / batchSize;
  }

  if (verbose) {
    solver->printStats();
    cout << "sparse elim ranges: " << printVec(solver->elimLumpRanges) << endl;
  }

  BenchResults retv;
  retv.analysisTime = analysisTime;
  retv.factorTime = factorTime;
  retv.solveTimes = solveTimes;
  return retv;
}
#endif  // BASPACHO_USE_CUBLAS

SparseProblem matGenToSparseProblem(SparseMatGenerator& gen, int64_t pSizeMin,
                                    int64_t pSizeMax) {
  SparseProblem retv;
  retv.sparseStruct = columnsToCscStruct(gen.columns).transpose();
  if (pSizeMin == pSizeMax) {
    retv.paramSize.assign(gen.columns.size(), pSizeMin);
  } else {
    retv.paramSize = randomVec(gen.columns.size(), pSizeMin, pSizeMax, gen.gen);
  }
  return retv;
}

map<string, function<SparseProblem(int64_t)>> problemGenerators = {

    // random entries
    {"10_FLAT_size=1000_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"11_FLAT_size=4000_fill=0.01_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen = SparseMatGenerator::genFlat(4000, 0.01, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"12_FLAT_size=2000_fill=0.03_bsize=2-5",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen = SparseMatGenerator::genFlat(2000, 0.03, seed);
       return matGenToSparseProblem(gen, 2, 5);
     }},  //

    // random entries + schur
    {"20_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=50000_schurfill=0.02",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
       gen.addSchurSet(50000, 0.02);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"21_FLAT+SCHUR_size=1000_fill=0.1_bsize=3_schursize=5000_schurfill=0.2",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen = SparseMatGenerator::genFlat(1000, 0.1, seed);
       gen.addSchurSet(5000, 0.0002);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //

    // base is grid
    {"30_GRID_size=100x100_fill=1.0_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genGrid(100, 100, 1.0, 2, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"31_GRID_size=150x150_fill=1.0_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genGrid(150, 150, 1.0, 2, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"32_GRID_size=200x200_fill=0.25_conn=2_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genGrid(200, 200, 0.25, 2, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"33_GRID_size=200x200_fill=0.05_conn=3_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genGrid(150, 150, 0.05, 3, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //

    // base is meridians
    {"40_MERI_size=1500_n=4_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genMeridians(4, 1500, 0.5, 120, 600, 2, 2, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
    {"41_MERI_size=1500_n=7_hairlen=600_hairs=2_band=120_fill=0.1_bsize=3",
     [](int64_t seed) -> SparseProblem {
       SparseMatGenerator gen =
           SparseMatGenerator::genMeridians(7, 1500, 0.5, 120, 600, 2, 2, seed);
       return matGenToSparseProblem(gen, 3, 3);
     }},  //
};

map<string, function<BenchResults(const SparseProblem&,
                                  const std::vector<int64_t>& nRHSs, bool)>>
    solvers = {
#ifdef BASPACHO_HAVE_CHOLMOD
        {"1_CHOLMOD",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           auto result = benchmarkCholmodSolve(
               prob.paramSize, prob.sparseStruct, nRHSs, verbose);
           BenchResults retv;
           retv.analysisTime = result.analysisTime;
           retv.factorTime = result.factorTime;
           retv.solveTimes = result.solveTimes;
           return retv;
         }},
#endif  // BASPACHO_HAVE_CHOLMOD
        {"2_BaSpaCho_BLAS_numthreads=16",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           return benchmarkSolver(prob, {}, nRHSs, verbose);
         }},  //
#ifdef BASPACHO_USE_CUBLAS
        {"3_BaSpaCho_CUDA",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           return benchmarkSolver(
               prob,
               {.findSparseEliminationRanges = true, .backend = BackendCuda},
               nRHSs, verbose);
         }},
        {"4_BaSpaCho_CUDA_batchsize=4",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           return benchmarkSolverBatched(
               prob,
               {.findSparseEliminationRanges = true, .backend = BackendCuda},
               /* batchsize = */ 4, nRHSs, verbose);
         }},
        {"5_BaSpaCho_CUDA_batchsize=8",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           return benchmarkSolverBatched(
               prob,
               {.findSparseEliminationRanges = true, .backend = BackendCuda},
               /* batchsize = */ 8, nRHSs, verbose);
         }},
        {"6_BaSpaCho_CUDA_batchsize=16",
         [](const SparseProblem& prob, const std::vector<int64_t>& nRHSs,
            bool verbose) -> BenchResults {
           return benchmarkSolverBatched(
               prob,
               {.findSparseEliminationRanges = true, .backend = BackendCuda},
               /* batchsize = */ 16, nRHSs, verbose);
         }},
#endif  // BASPACHO_USE_CUBLAS
};

struct BenchmarkSettings {
  int numIterations = 5;
  int verbose = true;
  regex selectProblems = regex("");
  regex excludeProblems;
  regex selectSolvers = regex("");
  regex excludeSolvers;
  string referenceSolver;
  vector<int64_t> nRHSs;
  set<string> operations = {"factor"};
};

static string ANALYSIS = "analysis";
static string FACTOR = "factor";
static string SOLVE_ = "solve-";
map<string, string> timingLabels = {
    {ANALYSIS, "Symbolic analysis and precomputing of needed indices"},
    {FACTOR, "Numeric computation of matrix factorization"},
    {SOLVE_ + "X", "Solve Ax=b with b having X columns (nRHS=X)"},
};

void runBenchmarks(const BenchmarkSettings& settings, int seed = 37) {
  if (!settings.referenceSolver.empty()) {
    if (solvers.find(settings.referenceSolver) == solvers.end()) {
      std::cerr << "Solver '" << settings.referenceSolver
                << "' does not exist or not available at compile time"
                << std::endl;
      exit(1);
    }
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

    cout << "\nProblem type: " << probName << endl;

    map<string, map<string, vector<double>>> timingSets;
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
           << "(" << it + 1 << "/" << settings.numIterations << ", " << solvName
           << " " << solvIdx << "/" << numSolversToTest << ")";
        solvIdx++;
        string str = ss.str();
        int clearSize = max(0, prevLen - (int)str.size());
        prevLen = str.size();
        cout << "\r" << str << setfill(' ') << setw(clearSize) << "" << flush;
        if (settings.verbose) {
          cout << endl;
        }

        auto benchResults = solv(prob, settings.nRHSs, settings.verbose);

        if (settings.verbose) {
          stringstream ss;
          ss << "analysis: " << benchResults.analysisTime
             << "s, factor: " << benchResults.factorTime << "s";
          for (auto [nRHS, timing] : benchResults.solveTimes) {
            ss << ", solve-" << nRHS << ": " << timing << "s";
          }
          cout << ss.str() << endl << endl;
        }

        timingSets[ANALYSIS][solvName].push_back(benchResults.analysisTime);
        timingSets[FACTOR][solvName].push_back(benchResults.factorTime);
        for (auto [nRHS, timing] : benchResults.solveTimes) {
          timingSets[SOLVE_ + to_string(nRHS)][solvName].push_back(timing);
        }
      }
    }
    stringstream ss;
    ss << setfill('.') << setw(settings.numIterations) << ""
       << "(" << settings.numIterations << "/" << settings.numIterations
       << ", done!)";
    string str = ss.str();
    int clearSize = max(0, prevLen - (int)str.size());
    cout << "\r" << str << setfill(' ') << setw(clearSize) << "" << endl;

    for (auto [label, solverTimings] : timingSets) {
      if (settings.operations.find(label) == settings.operations.end()) {
        continue;
      }
      cout << "Operation: " << label << endl;
      auto it = solverTimings.find(settings.referenceSolver);
      const vector<double>* refTimings =
          (it != solverTimings.end()) ? &it->second : nullptr;
      if (refTimings) {
        auto& timings = *refTimings;
        stringstream ss;
        ss << "- " << settings.referenceSolver
           << " (basis for comparison):\n    ";
        for (size_t i = 0; i < timings.size(); i++) {
          stringstream tss;
          if (timings[i] > 0.1) {
            tss << fixed << setprecision(3) << timings[i] << "s";
          } else {
            tss << fixed << setprecision(1) << timings[i] * 1000 << "ms";
          }
          tss << (i == timings.size() - 1 ? "" : ", ");
          ss << left << setfill(' ') << setw(20) << tss.str();
        }
        cout << ss.str() << endl;
      }
      for (auto [solvName, timings] : solverTimings) {
        if (solvName == settings.referenceSolver) {
          continue;
        }

        stringstream ss;
        ss << "- " << solvName;
        if (refTimings) {
          ss << " (vs. " << settings.referenceSolver << ")";
        }
        ss << ":\n    ";
        for (size_t i = 0; i < timings.size(); i++) {
          stringstream tss;
          if (timings[i] > 0.1) {
            tss << fixed << setprecision(3) << timings[i] << "s";
          } else {
            tss << fixed << setprecision(1) << timings[i] * 1000 << "ms";
          }
          if (refTimings) {
            double percent = (timings[i] / (*refTimings)[i] - 1.0) * 100.0;
            tss << " (" << (percent > 0 ? "+" : "") << fixed << setprecision(2)
                << percent << "%)";
          }
          tss << (i == timings.size() - 1 ? "" : ", ");
          ss << left << setfill(' ') << setw(20) << tss.str();
        }
        cout << ss.str() << endl;
      }
    }
  }
}

void help() {
  cout << "This program runs a benchmark of several solver configurations"
       << "\non different synthetic problem types, printing timings and"
       << "\nrelative timings for different operations"
       << "\n -v           [v]erbose stats for factor/solve runs"
       << "\n -n number    [n]umber of problem per type (default: 5)"
       << "\n -S regex     regex for selecting [S]olver types"
       << "\n -E regex     regex for [E]xcluding solver types"
       << "\n -R regex     [R]egex for selecting problem types"
       << "\n -X regex     regex for e[X]cluding problem types"
       << "\n -B solver    solver selected as [B]baseline"
       << "\n -O ops       comma-sep list of operations (default: factor)"
       << endl;

  cout << "\nOperations:" << endl;
  for (const auto& [label, desc] : timingLabels) {
    cout << "  " << left << setfill(' ') << setw(15) << label << desc << endl;
  }
  cout << "\nSolvers:" << endl;
  for (const auto& [solvName, solv] : solvers) {
    cout << "  " << solvName << endl;
  }
  cout << "\nProblem generators:" << endl;
  for (const auto& [probName, gen] : problemGenerators) {
    cout << "  " << probName << endl;
  }
}

int main(int argc, char* argv[]) {
  BenchmarkSettings settings;
  // settings.referenceSolver = "CHOLMOD";
  settings.verbose = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      help();
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
    } else if (!strcmp(argv[i], "-B") && i < argc - 1) {
      settings.referenceSolver = argv[++i];
    } else if (!strcmp(argv[i], "-O") && i < argc - 1) {
      stringstream ss(argv[++i]);
      set<string> operations;
      set<int64_t> nRHSs;

      while (ss.good()) {
        string substr;
        getline(ss, substr, ',');
        if (substr.rfind(SOLVE_, 0) == 0) {
          std::string arg = substr.substr(SOLVE_.size());
          try {
            size_t endpos;
            int64_t nRHS = stoi(arg, &endpos);
            if (nRHS < 1 || endpos != arg.size()) {
              throw exception();
            }
            nRHSs.insert(nRHS);
            operations.insert(SOLVE_ + to_string(nRHS));
          } catch (...) {
            cerr << "Invalid argument '" << arg << "' for '" << SOLVE_
                 << "' option, must be an integer >=1" << endl;
            return 1;
          }
          continue;
        }
        if (timingLabels.find(substr) == timingLabels.end()) {
          cerr << "Operation '" << substr << "' does not exist! (-h)" << endl;
          return 1;
        }
        operations.insert(substr);
      }
      settings.nRHSs.assign(nRHSs.begin(), nRHSs.end());
      settings.operations = operations;
    }
  }

  runBenchmarks(settings);

  return 0;
}