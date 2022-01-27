
#include "Utils.h"

#include <glog/logging.h>

#include <sstream>

using namespace std;

using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

string OpStat::toString() const {
    stringstream ss;
    ss << "#=" << numRuns << ", time=" << totTime << "s, last=" << lastTime
       << "s, max=" << maxTime << "s";
    return ss.str();
}

OpInstance::OpInstance(OpStat& stat) : stat(stat), start(hrc::now()) {}

OpInstance::~OpInstance() {
    stat.numRuns++;
    stat.lastTime = tdelta(hrc::now() - start).count();
    stat.maxTime = max(stat.maxTime, stat.lastTime);
    stat.totTime += stat.lastTime;
}

std::vector<uint64_t> composePermutations(const std::vector<uint64_t>& v,
                                          const std::vector<uint64_t>& w) {
    CHECK_EQ(v.size(), w.size());
    std::vector<uint64_t> retv(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        retv[i] = v[w[i]];
    }
    return retv;
}

std::vector<uint64_t> inversePermutation(const std::vector<uint64_t>& p) {
    std::vector<uint64_t> retv(p.size());
    for (size_t i = 0; i < p.size(); i++) {
        retv[p[i]] = i;
    }
    return retv;
}

uint64_t cumSumVec(vector<uint64_t>& v) {
    uint64_t numEls = v.size() - 1;
    uint64_t tot = 0;
    for (uint64_t i = 0; i < numEls; i++) {
        uint64_t oldTot = tot;
        tot += v[i];
        v[i] = oldTot;
    }
    v[numEls] = tot;
    return tot;
}

void rewindVec(std::vector<uint64_t>& v, uint64_t downTo, uint64_t value) {
    for (uint64_t i = v.size() - 1; i > downTo; i--) {
        v[i] = v[i - 1];
    }
    v[downTo] = value;
}