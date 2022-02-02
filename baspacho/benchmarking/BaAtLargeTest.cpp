
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
    std::cout << "tot cost: " << totCost << std::endl;
}

void testSolvers(Data& data) {
    int64_t numPts = data.points.size();
    int64_t numCams = data.cameras.size();
    int64_t totNumParams = numPts + numCams;

    std::cout << "Build struct..." << std::endl;
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

    std::cout << "Building ss..." << std::endl;
    SparseStructure origSs = columnsToCscStruct(colBlocks).transpose();

    std::cout << "===========================================" << std::endl;
    std::cout << "Testing BLAS..." << std::endl;
    {
        auto solver = createSolverSchur({}, paramSize, origSs, {0, numPts});

        // generate mock data, make positive def
        vector<double> matData =
            randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
        solver->factorSkel.damp(matData, 0, solver->factorSkel.order() * 1.2);

        auto startFactor = hrc::now();
        solver->factor(matData.data());
        double factorTime = tdelta(hrc::now() - startFactor).count();

        solver->printStats();
        std::cout << "Total Factor: " << factorTime << endl << std::endl;
    }

#if defined(BASPACHO_USE_CUBLAS) || defined(BASPACHO_HAVE_CHOLMOD)
    // create cam-cam system
    vector<int64_t> camSz(numCams, 6);
    SparseStructure elimPtSs = origSs.addIndependentEliminationFill(0, numPts);
    SparseStructure camCamSs = elimPtSs.extractRightBottom(numPts);
#endif  // defined(BASPACHO_USE_CUBLAS) || defined(BASPACHO_HAVE_CHOLMOD)

    // test Cuda
#ifdef BASPACHO_USE_CUBLAS
    std::cout << "===========================================" << std::endl;
    std::cout << "Testing CUDA..." << std::endl;
    {
        auto solver = createSolver(
            {.findSparseEliminationRanges = false, .backend = BackendCuda},
            camSz, camCamSs);

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
        std::cout << "Total Factor: " << factorTime << endl << std::endl;
    }
#endif  // BASPACHO_USE_CUBLAS

    // test Cholmod
#ifdef BASPACHO_HAVE_CHOLMOD
    std::cout << "===========================================" << std::endl;
    std::cout << "Testing CHOLMOD..." << std::endl;
    auto [aTime, fTime] = benchmarkCholmodSolve(camSz, camCamSs, true);
    std::cout << "Cholmod Analysis Time: " << aTime << "s" << std::endl;
    std::cout << "Cholmod Factor Time: " << fTime << "s" << std::endl;
#endif  // BASPACHO_HAVE_CHOLMOD
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        std::cout << "Usage: prog bal_file.txt" << std::endl;
        return 1;
    }

    Data data;
    data.load(argv[1], true);

    testSolvers(data);

    return 0;
}