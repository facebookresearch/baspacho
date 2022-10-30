/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/EliminationTree.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/SparseStructure.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/benchmarking/BaAtLarge.h"
#include "baspacho/testing/TestingUtils.h"

#ifdef BASPACHO_USE_CUBLAS
#include "baspacho/baspacho/CudaDefs.h"
#endif

#ifdef BASPACHO_HAVE_CHOLMOD
#include "BenchCholmod.h"
#endif

using namespace BaSpaCho;
using namespace testing_utils;
using namespace ba_at_large;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

void computeCost(Data& data) {
  double totCost = 0;
  for (auto& obs : data.observations) {
    auto& cam = data.cameras[obs.camIdx];
    auto& pt = data.points[obs.ptIdx];
    Vec2 err = Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2);
    totCost += 0.5 * err.squaredNorm();
  }
  cout << "tot cost: " << totCost << endl;
}

void testSolvers(Data& data, int nPointParams, int nCameraParams) {
  int64_t numPts = data.points.size();
  int64_t numCams = data.cameras.size();
  int64_t totNumParams = numPts + numCams;

  // cout << "Build struct..." << endl;
  vector<int64_t> paramSize(totNumParams);
  vector<set<int64_t>> colBlocks(totNumParams);
  for (int64_t i = 0; i < numPts; i++) {  // points go first
    paramSize[i] = nPointParams;
    colBlocks[i].insert(i);
  }
  for (int64_t i = numPts; i < totNumParams; i++) {  // then cams
    paramSize[i] = nCameraParams;
    colBlocks[i].insert(i);
  }
  for (auto& obs : data.observations) {
    colBlocks[obs.ptIdx].insert(numPts + obs.camIdx);
  }

  // cout << "Building ss..." << endl;
  SparseStructure origSs = columnsToCscStruct(colBlocks).transpose();

  if (1) {
    cout << "===========================================" << endl;
    cout << "Testing Baspacho/BLAS nThreads=16 (on full Points+Cameras "
            "system)"
         << endl;
    auto startAnalysis = hrc::now();
    auto solver = createSolver({.numThreads = 16}, paramSize, origSs, {0, numPts});
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // generate mock data, make positive def
    vector<double> matData = randomData(solver->dataSize(), -1.0, 1.0, 37);
    solver->skel().damp(matData, double(0), double(solver->order() * 1.2));

    cout << "heating up factor..." << endl;
    solver->factor(matData.data());  // heat up

    solver->enableStats();
    solver->resetStats();
    cout << "running real benchmark..." << endl;

    auto startFactor = hrc::now();
    solver->factor(matData.data());
    double factorTime = tdelta(hrc::now() - startFactor).count();

    solver->printStats();
    double elimTime = solver->internalGetElimCtx(0).elimStat.totTime;
    cout << "Total Analysis Time..: " << secondsToString(analysisTime) << endl;
    cout << "Total Factor Time....: " << secondsToString(factorTime) << endl;
    cout << "Point Schur-Elim Time: " << secondsToString(elimTime) << endl;
    cout << "Cam-Cam Factor Time..: " << secondsToString(factorTime - elimTime) << endl << endl;
  }

  if (1) {
    cout << "===========================================" << endl;
    cout << "Testing Baspacho/BLAS nThreads=1 (on full Points+Cameras "
            "system)"
         << endl;
    auto startAnalysis = hrc::now();
    auto solver = createSolver({.numThreads = 1}, paramSize, origSs, {0, numPts});
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // generate mock data, make positive def
    vector<double> matData = randomData(solver->dataSize(), -1.0, 1.0, 37);
    solver->skel().damp(matData, double(0), double(solver->order() * 1.2));

    cout << "heating up factor..." << endl;
    solver->factor(matData.data());  // heat up

    solver->enableStats();
    solver->resetStats();
    cout << "running real benchmark..." << endl;

    auto startFactor = hrc::now();
    solver->factor(matData.data());
    double factorTime = tdelta(hrc::now() - startFactor).count();

    solver->printStats();
    double elimTime = solver->internalGetElimCtx(0).elimStat.totTime;
    cout << "Total Analysis Time..: " << secondsToString(analysisTime) << endl;
    cout << "Total Factor Time....: " << secondsToString(factorTime) << endl;
    cout << "Point Schur-Elim Time: " << secondsToString(elimTime) << endl;
    cout << "Cam-Cam Factor Time..: " << secondsToString(factorTime - elimTime) << endl << endl;
  }

#if defined(BASPACHO_USE_CUBLAS) || defined(BASPACHO_HAVE_CHOLMOD)
  // create cam-cam system
  vector<int64_t> camSz(numCams, nCameraParams);
  SparseStructure elimPtSs = origSs.addIndependentEliminationFill(0, numPts);
  SparseStructure camCamSs = elimPtSs.extractRightBottom(numPts);
#endif  // defined(BASPACHO_USE_CUBLAS) || defined(BASPACHO_HAVE_CHOLMOD)

  // test Cuda
#ifdef BASPACHO_USE_CUBLAS
  {
    cout << "===========================================" << endl;
    cout << "Testing CUDA (on full Points+Cameras system)" << endl;
    {
      cout << "heating up cuda..." << endl;
      auto solver = createSolver({.backend = BackendCuda}, paramSize, origSs, {0, numPts});
    }
    auto startAnalysis = hrc::now();
    auto solver = createSolver({.backend = BackendCuda}, paramSize, origSs, {0, numPts});
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // generate mock data, make positive def
    vector<double> matData = randomData(solver->dataSize(), -1.0, 1.0, 37);
    solver->skel().damp(matData, double(0), double(solver->order() * 1.2));

    {
      cout << "heating up factor..." << endl;
      DevMirror<double> matDataGpu(matData);
      solver->factor(matDataGpu.ptr);
      cout << "running real benchmark..." << endl;
    }

    solver->enableStats();
    solver->resetStats();

    double factorTime;
    {
      DevMirror<double> matDataGpu(matData);
      auto startFactor = hrc::now();
      solver->factor(matDataGpu.ptr);
      factorTime = tdelta(hrc::now() - startFactor).count();
    }

    solver->printStats();
    double elimTime = solver->internalGetElimCtx(0).elimStat.totTime;
    cout << "Total Analysis Time..: " << secondsToString(analysisTime) << endl;
    cout << "Total Factor Time....: " << secondsToString(factorTime) << endl;
    cout << "Point Schur-Elim Time: " << secondsToString(elimTime) << endl;
    cout << "Cam-Cam Factor Time..: " << secondsToString(factorTime - elimTime) << endl << endl;
  }
#endif  // BASPACHO_USE_CUBLAS

  // test Cuda
#ifdef BASPACHO_USE_CUBLAS
  {
    cout << "===========================================" << endl;
    cout << "Testing CUDA (on reduced Camera-Camera matrix)" << endl;
    auto startAnalysis = hrc::now();
    auto solver = createSolver({.findSparseEliminationRanges = false, .backend = BackendCuda},
                               camSz, camCamSs);
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();

    // generate mock data, make positive def
    vector<double> matData = randomData(solver->dataSize(), -1.0, 1.0, 37);
    solver->skel().damp(matData, double(0), double(solver->order() * 1.2));

    {
      cout << "heating up factor..." << endl;
      DevMirror<double> matDataGpu(matData);
      solver->factor(matDataGpu.ptr);
      cout << "running real benchmark..." << endl;
    }

    solver->enableStats();
    solver->resetStats();

    double factorTime;
    {
      DevMirror<double> matDataGpu(matData);
      auto startFactor = hrc::now();
      solver->factor(matDataGpu.ptr);
      factorTime = tdelta(hrc::now() - startFactor).count();
    }

    solver->printStats();
    cout << "Cam-Cam Analysis Time: " << secondsToString(analysisTime) << endl;
    cout << "Cam-Cam Factor Time..: " << secondsToString(factorTime) << endl << endl;
  }
#endif  // BASPACHO_USE_CUBLAS

  // test Cholmod
#ifdef BASPACHO_HAVE_CHOLMOD
  {
    cout << "===========================================" << endl;
    cout << "Testing CHOLMOD (on reduced Camera-Camera matrix)" << endl;
    {
      cout << "heating up factor..." << endl;
      benchmarkCholmodSolve(camSz, camCamSs, {1, 10}, false);
      cout << "running real benchmark..." << endl;
    }
    auto results = benchmarkCholmodSolve(camSz, camCamSs, {1, 10}, true);
    cout << "Cam-Cam Analysis Time: " << secondsToString(results.analysisTime) << endl;
    cout << "Cam-Cam Factor Time..: " << secondsToString(results.factorTime) << endl;
    for (auto [nRHS, timing] : results.solveTimes) {
      cout << "Cam-Cam Solve-" << nRHS << " Time.: " << secondsToString(timing) << endl;
    }
  }
#endif  // BASPACHO_HAVE_CHOLMOD
}

void help() {
  cout << "BAL_bench -i file.txt\n"
       << " -c    num of [c]amera parameters (default: 9)\n"
       << " -p    num of [p]oint parameters (default: 3)" << std::endl;
}

int main(int argc, char* argv[]) {
  int nPointParams = 3;
  int nCameraParams = 9;
  std::string input;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h")) {
      help();
      return 0;
    }
    if (!strcmp(argv[i], "-i") && i < argc - 1) {
      input = argv[++i];
    } else if (!strcmp(argv[i], "-c") && i < argc - 1) {
      nCameraParams = std::stoi(argv[++i]);
    } else if (!strcmp(argv[i], "-p") && i < argc - 1) {
      nPointParams = std::stoi(argv[++i]);
    } else {
      std::cout << "Error: unknown argument: '" << argv[i] << "'" << std::endl;
      help();
      return 0;
    }
  }
  if (input.empty()) {
    std::cout << "Error: missing input file (-i file.txt)" << std::endl;
    help();
    return 0;
  }

  if (argc <= 1) {
    cout << "Usage: prog bal_file.txt" << endl;
    return 1;
  }

  cout << "Loading data..." << endl;
  Data data;
  data.load(input, false);

  testSolvers(data, nPointParams, nCameraParams);

  return 0;
}