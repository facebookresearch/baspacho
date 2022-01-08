
#include <glog/logging.h>

#include "BaAtLarge.h"
#include "BlockStructure.h"

using namespace ba_at_large;

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
    std::vector<uint64_t> paramSize(totNumParams);
    std::vector<std::set<uint64_t>> colBlocks(totNumParams);
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

    LOG(INFO) << "Elim pts...";
    BlockStructure blockStructure(paramSize, colBlocks);
    blockStructure.addBlocksForEliminationOfRange(0, numPts);

    uint64_t totPossible = numCams * (numCams + 1) / 2;

    uint64_t camCamBlocks =
        blockStructure.numBlocksInCols(numPts, totNumParams);
    LOG(INFO) << "cam-cam blocks (from pts): " << camCamBlocks << " ("
              << (100.0 * camCamBlocks / totPossible) << "%)";

    // blockStructure.applyAmdFrom(numPts);

    blockStructure.addBlocksForEliminationOfRange(numPts, totNumParams);
    uint64_t camCamBlocks2 =
        blockStructure.numBlocksInCols(numPts, totNumParams);
    LOG(INFO) << "cam-cam blocks (with fill): " << camCamBlocks2 << " ("
              << (100.0 * camCamBlocks2 / totPossible) << "%)";
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