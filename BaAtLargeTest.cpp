
#include <glog/logging.h>

#include "BaAtLarge.h"

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

void experiment(Data& data) {}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LOG(INFO) << "Usage: prog bal_file.txt";
        return 1;
    }

    Data data;
    data.load(argv[1], true);

    computeCost(data);

    return 0;
}