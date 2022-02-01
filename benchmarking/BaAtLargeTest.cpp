
#include <chrono>

#include "../baspacho/CoalescedBlockMatrix.h"
#include "../baspacho/DebugMacros.h"
#include "../baspacho/EliminationTree.h"
#include "../baspacho/Solver.h"
#include "../baspacho/SparseStructure.h"
#include "../baspacho/Utils.h"
#include "../testing/TestingUtils.h"
#include "BaAtLarge.h"

#ifdef BASPACHO_HAVE_CHOLMOD
#include "BenchCholmod.h"
#endif

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

void experiment2(Data& data) {
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

    bool verbose = 1;
    auto solver = createSolverSchur({}, paramSize, origSs,
                                    vector<int64_t>{0, numPts}, verbose);

    // generate mock data, make positive def
    vector<double> matData =
        randomData(solver->factorSkel.dataSize(), -1.0, 1.0, 37);
    solver->factorSkel.damp(matData, 0, solver->factorSkel.order() * 1.2);

    auto startFactor = hrc::now();
    solver->factor(matData.data(), verbose);
    double factorTime = tdelta(hrc::now() - startFactor).count();

    if (verbose) {
        // solver->ops->printStats(); // FIXME
        if (verbose) {
            std::cout << "factor: " << factorTime << endl << std::endl;
        }
    }
}

void experiment(Data& data) {
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

    std::cout << "Elim pts..." << std::endl;
    SparseStructure elimPtSs = origSs.addIndependentEliminationFill(0, numPts);
    std::cout << "done!" << std::endl;

    SparseStructure elimPtSsT = elimPtSs.transpose();
    int64_t totPossible = numCams * (numCams + 1) / 2;
    int64_t camCamBlocks =
        elimPtSsT.ptrs[totNumParams] - elimPtSsT.ptrs[numPts];
    std::cout << "cam-cam blocks (from pts): " << camCamBlocks << " ("
              << (100.0 * camCamBlocks / totPossible) << "%)" << std::endl;

    SparseStructure camCamSs = elimPtSs.extractRightBottom(numPts);

#if 1
    std::cout << "Applying permutation..." << std::endl;
    vector<int64_t> permutation = camCamSs.fillReducingPermutation();
    vector<int64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedCamCamSs =
        camCamSs.symmetricPermutation(invPerm, false).addFullEliminationFill();
#else
    SparseStructure sortedCamCamSs = camCamSs;
#endif

    std::cout << "Computing remaining fill in..." << std::endl;
    SparseStructure filledSs = sortedCamCamSs.addFullEliminationFill();
    std::cout << "done!" << std::endl;

    SparseStructure filledSsT = filledSs.transpose();
    int64_t camCamBlocks2 = filledSsT.ptrs[numCams];
    std::cout << "cam-cam blocks (with fill): " << camCamBlocks2 << " ("
              << (100.0 * camCamBlocks2 / totPossible) << "%)" << std::endl;

    std::cout << "Computing elim tree" << std::endl;

    vector<int64_t> paramSz(sortedCamCamSs.ptrs.size() - 1, 6);
    EliminationTree et(paramSz, sortedCamCamSs);

    std::cout << "Build tree" << std::endl;
    et.buildTree();

    std::cout << "Merges" << std::endl;
    et.computeMerges();

    std::cout << "Aggreg" << std::endl;
    et.computeAggregateStruct();

    std::cout << "Block mat" << std::endl;
    CoalescedBlockMatrixSkel factorSkel(et.spanStart, et.lumpToSpan,
                                        et.colStart, et.rowParam);
    int64_t totData = factorSkel.dataSize();
    std::cout << "cam-cam blocky (with fill): " << totData << " ("
              << (100.0 * totData / (numCams * numCams)) << "%)" << std::endl;

    std::cout << "aggregBlocks:" << factorSkel.lumpToSpan.size() - 1
              << std::endl;
    for (size_t a = 0; a < factorSkel.lumpToSpan.size() - 1; a++) {
        std::cout
            << "a." << a << ": size="
            << factorSkel.lumpToSpan[a + 1] - factorSkel.lumpToSpan[a]
            << ", nBlockRows="
            << factorSkel.boardChainColOrd[factorSkel.boardColPtr[a + 1] - 1]
            << std::endl;
    }

#ifdef BASPACHO_HAVE_CHOLMOD
    std::cout << "Testing Cholmod..." << std::endl;
    benchmarkCholmodSolve(paramSz, camCamSs);
#endif
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        std::cout << "Usage: prog bal_file.txt" << std::endl;
        return 1;
    }

    Data data;
    data.load(argv[1], true);

    // computeCost(data);
    experiment2(data);

    return 0;
}