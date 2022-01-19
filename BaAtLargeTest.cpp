
#include <glog/logging.h>

#include "BaAtLarge.h"
#include "BenchCholmod.h"
#include "BlockMatrix.h"
#include "EliminationTree.h"
#include "SparseStructure.h"
#include "TestingUtils.h"
#include "Utils.h"

using namespace ba_at_large;
using namespace std;

void computeCost(Data& data) {
    double totCost = 0;
    for (auto& obs : data.observations) {
        auto& cam = data.cameras[obs.camIdx];
        auto& pt = data.points[obs.ptIdx];
        Vec2 err =
            Cost::compute_residual(obs.imgPos, pt, cam.T_W_C, cam.f_k1_k2);
        totCost += 0.5 * err.squaredNorm();
    }
    LOG(INFO) << "tot cost: " << totCost;
}

void experiment(Data& data) {
    uint64_t numPts = data.points.size();
    uint64_t numCams = data.cameras.size();
    uint64_t totNumParams = numPts + numCams;

    LOG(INFO) << "Build struct...";
    vector<uint64_t> paramSize(totNumParams);
    vector<set<uint64_t>> colBlocks(totNumParams);
    for (uint64_t i = 0; i < numPts; i++) {  // points go first
        paramSize[i] = 3;
        colBlocks[i].insert(i);
    }
    for (uint64_t i = numPts; i < totNumParams; i++) {  // then cams
        paramSize[i] = 6;
        colBlocks[i].insert(i);
    }
    for (auto& obs : data.observations) {
        colBlocks[obs.ptIdx].insert(numPts + obs.camIdx);
    }

    LOG(INFO) << "Building ss...";
    SparseStructure origSs = columnsToCscStruct(colBlocks).transpose();

    LOG(INFO) << "Elim pts...";
    SparseStructure elimPtSs = origSs.addIndependentEliminationFill(0, numPts);
    LOG(INFO) << "done!";

    SparseStructure elimPtSsT = elimPtSs.transpose();
    uint64_t totPossible = numCams * (numCams + 1) / 2;
    uint64_t camCamBlocks =
        elimPtSsT.ptrs[totNumParams] - elimPtSsT.ptrs[numPts];
    LOG(INFO) << "cam-cam blocks (from pts): " << camCamBlocks << " ("
              << (100.0 * camCamBlocks / totPossible) << "%)";

    SparseStructure camCamSs = elimPtSs.extractRightBottom(numPts);

#if 1
    LOG(INFO) << "Applying permutation...";
    vector<uint64_t> permutation = camCamSs.fillReducingPermutation();
    vector<uint64_t> invPerm = inversePermutation(permutation);
    SparseStructure sortedCamCamSs =
        camCamSs.symmetricPermutation(invPerm, false).addFullEliminationFill();
#else
    SparseStructure sortedCamCamSs = camCamSs;
#endif

    LOG(INFO) << "Computing remaining fill in...";
    SparseStructure filledSs = sortedCamCamSs.addFullEliminationFill();
    LOG(INFO) << "done!";

    SparseStructure filledSsT = filledSs.transpose();
    uint64_t camCamBlocks2 = filledSsT.ptrs[numCams];
    LOG(INFO) << "cam-cam blocks (with fill): " << camCamBlocks2 << " ("
              << (100.0 * camCamBlocks2 / totPossible) << "%)";

    LOG(INFO) << "Computing elim tree";

    vector<uint64_t> paramSz(sortedCamCamSs.ptrs.size() - 1, 6);
    EliminationTree et(paramSz, sortedCamCamSs);

    LOG(INFO) << "Build tree";
    et.buildTree();

    LOG(INFO) << "Merges";
    et.computeMerges();

    LOG(INFO) << "Aggreg";
    et.computeAggregateStruct();

    LOG(INFO) << "Block mat";
    BlockMatrixSkel skel(et.spanStart, et.rangeToSpan, et.colStart,
                         et.rowParam);
    uint64_t totData = skel.sliceData[skel.sliceData.size() - 1];
    LOG(INFO) << "cam-cam blocky (with fill): " << totData << " ("
              << (100.0 * totData / (numCams * numCams)) << "%)";

    LOG(INFO) << "aggregBlocks:" << skel.rangeToSpan.size() - 1;
    for (size_t a = 0; a < skel.rangeToSpan.size() - 1; a++) {
        LOG(INFO) << "a." << a
                  << ": size=" << skel.rangeToSpan[a + 1] - skel.rangeToSpan[a]
                  << ", nBlockRows="
                  << skel.slabSliceColOrd[skel.slabColPtr[a + 1] - 1];
    }

    LOG(INFO) << "Testing Cholmod...";
    benchmarkCholmodSolve(paramSz, camCamSs);
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LOG(INFO) << "Usage: prog bal_file.txt";
        return 1;
    }

    Data data;
    data.load(argv[1], true);

    // computeCost(data);
    experiment(data);

    return 0;
}