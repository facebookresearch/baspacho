
#include <chrono>

#include "baspacho/CoalescedBlockMatrix.h"
#include "baspacho/DebugMacros.h"
#include "baspacho/EliminationTree.h"
#include "baspacho/Solver.h"
#include "baspacho/SparseStructure.h"
#include "baspacho/Utils.h"
#include "baspacho/benchmarking/BaAtLarge.h"
#include "baspacho/testing/TestingUtils.h"

#ifdef BASPACHO_USE_CUBLAS
#include "baspacho/CudaDefs.h"
#endif

#ifdef BASPACHO_HAVE_CHOLMOD
#include "BenchCholmod.h"
#endif

using namespace BaSpaCho;
using namespace testing;
using namespace ba_at_large;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

void computeCost(Data& data) {
    double totCost = 0;
    for (auto& obs : data.observations) {
        auto& cam = data.cameras[obs.camIdx];
        auto& pt = data.points[obs.ptIdx];
        Vec2 err =
            Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2);
        totCost += 0.5 * err.squaredNorm();
    }
    cout << "tot cost: " << totCost << endl;
}

void testSolvers(Data& data) {
    int64_t numPts = data.points.size();
    int64_t numCams = data.cameras.size();
    int64_t totNumParams = numPts + numCams;

    // cout << "Build struct..." << endl;
    vector<int64_t> paramSize(totNumParams);
    vector<set<int64_t>> colBlocks(totNumParams);
    for (int64_t i = 0; i < numPts; i++) {  // points go first
        paramSize[i] = 3;
        colBlocks[i].insert(i);
    }
    for (int64_t i = numPts; i < totNumParams; i++) {  // then cams
        paramSize[i] = 6;
        colBlocks[i].insert(i);
    }
    for (auto& obs : data.observations) {
        colBlocks[obs.ptIdx].insert(numPts + obs.camIdx);
    }

    // cout << "Building ss..." << endl;
    SparseStructure origSs = columnsToCscStruct(colBlocks).transpose();

    if (0) {
        cout << "===========================================" << endl;
        cout << "Testing Baspacho/BLAS (on full Points+Cameras system)" << endl;
        auto startAnalysis = hrc::now();
        auto solver = createSolverSchur({}, paramSize, origSs, {0, numPts});
        double analysisTime = tdelta(hrc::now() - startAnalysis).count();

        // generate mock data, make positive def
        vector<double> matData =
            randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
        solver->factorSkel.damp(matData, 0, solver->factorSkel.order() * 1.2);

        auto startFactor = hrc::now();
        solver->factor(matData.data());
        double factorTime = tdelta(hrc::now() - startFactor).count();

        solver->printStats();
        double elimTime = solver->elimCtxs[0]->elimStat.totTime;
        cout << "Total Analysis Time..: " << analysisTime << "s" << endl;
        cout << "Total Factor Time....: " << factorTime << "s" << endl;
        cout << "Point Schur-Elim Time: " << elimTime << "s" << endl;
        cout << "Cam-Cam Factor Time..: " << factorTime - elimTime << "s"
             << endl
             << endl;
    }

#if defined(BASPACHO_USE_CUBLAS) || defined(BASPACHO_HAVE_CHOLMOD)
    // create cam-cam system
    vector<int64_t> camSz(numCams, 6);
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
            auto solver = createSolverSchur({.backend = BackendCuda}, paramSize,
                                            origSs, {0, numPts});
        }
        auto startAnalysis = hrc::now();
        auto solver = createSolverSchur({.backend = BackendCuda}, paramSize,
                                        origSs, {0, numPts});
        double analysisTime = tdelta(hrc::now() - startAnalysis).count();

        cout << "sparse elim ranges: " << printVec(solver->elimLumpRanges)
             << endl;

        // generate mock data, make positive def
        vector<double> matData =
            randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
        solver->factorSkel.damp(matData, 0, solver->factorSkel.order() * 1.2);

        double* dataGPU;
        cuCHECK(cudaMalloc((void**)&dataGPU, matData.size() * sizeof(double)));

        {
            cout << "heating up factor..." << endl;
            cuCHECK(cudaMemcpy(dataGPU, matData.data(),
                               matData.size() * sizeof(double),
                               cudaMemcpyHostToDevice));
            solver->factor(dataGPU);
            solver->resetStats();
        }

        cuCHECK(cudaMemcpy(dataGPU, matData.data(),
                           matData.size() * sizeof(double),
                           cudaMemcpyHostToDevice));
        auto startFactor = hrc::now();
        solver->factor(dataGPU);
        double factorTime = tdelta(hrc::now() - startFactor).count();
        cuCHECK(cudaMemcpy(matData.data(), dataGPU,
                           matData.size() * sizeof(double),
                           cudaMemcpyDeviceToHost));
        cuCHECK(cudaFree(dataGPU));

        solver->printStats();
        double elimTime = solver->elimCtxs[0]->elimStat.totTime;
        cout << "Total Analysis Time..: " << analysisTime << "s" << endl;
        cout << "Total Factor Time....: " << factorTime << "s" << endl;
        cout << "Point Schur-Elim Time: " << elimTime << "s" << endl;
        cout << "Cam-Cam Factor Time..: " << factorTime - elimTime << "s"
             << endl
             << endl;
    }
#endif  // BASPACHO_USE_CUBLAS

    // test Cuda
#ifdef BASPACHO_USE_CUBLAS
    {
        cout << "===========================================" << endl;
        cout << "Testing CUDA (on reduced Camera-Camera matrix)" << endl;
        auto startAnalysis = hrc::now();
        auto solver = createSolver(
            {.findSparseEliminationRanges = false, .backend = BackendCuda},
            camSz, camCamSs);
        double analysisTime = tdelta(hrc::now() - startAnalysis).count();

        // generate mock data, make positive def
        vector<double> matData =
            randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
        solver->factorSkel.damp(matData, 0, solver->factorSkel.order() * 1.2);

        double* dataGPU;
        cuCHECK(cudaMalloc((void**)&dataGPU, matData.size() * sizeof(double)));
        cuCHECK(cudaMemcpy(dataGPU, matData.data(),
                           matData.size() * sizeof(double),
                           cudaMemcpyHostToDevice));
        auto startFactor = hrc::now();
        solver->factor(dataGPU);
        double factorTime = tdelta(hrc::now() - startFactor).count();
        cuCHECK(cudaMemcpy(matData.data(), dataGPU,
                           matData.size() * sizeof(double),
                           cudaMemcpyDeviceToHost));
        cuCHECK(cudaFree(dataGPU));

        solver->printStats();
        cout << "Cam-Cam Analysis Time: " << analysisTime << "s" << endl;
        cout << "Cam-Cam Factor Time..: " << factorTime << "s" << endl << endl;
    }
#endif  // BASPACHO_USE_CUBLAS

    // test Cholmod
#ifdef BASPACHO_HAVE_CHOLMOD
    {
        cout << "===========================================" << endl;
        cout << "Testing CHOLMOD (on reduced Camera-Camera matrix)" << endl;
        auto [aTime, fTime] = benchmarkCholmodSolve(camSz, camCamSs, true);
        cout << "Cam-Cam Analysis Time: " << aTime << "s" << endl;
        cout << "Cam-Cam Factor Time..: " << fTime << "s" << endl;
    }
#endif  // BASPACHO_HAVE_CHOLMOD
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        cout << "Usage: prog bal_file.txt" << endl;
        return 1;
    }

    cout << "Loading data..." << endl;
    Data data;
    data.load(argv[1], false);

    testSolvers(data);

    return 0;
}