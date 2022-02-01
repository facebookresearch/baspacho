
#include "baspacho/Utils.h"

#include <sstream>

#include "baspacho/DebugMacros.h"

namespace BaSpaCho {

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

std::vector<int64_t> composePermutations(const std::vector<int64_t>& v,
                                         const std::vector<int64_t>& w) {
    BASPACHO_CHECK_EQ(v.size(), w.size());
    std::vector<int64_t> retv(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        retv[i] = v[w[i]];
    }
    return retv;
}

std::vector<int64_t> inversePermutation(const std::vector<int64_t>& p) {
    std::vector<int64_t> retv(p.size());
    for (size_t i = 0; i < p.size(); i++) {
        retv[p[i]] = i;
    }
    return retv;
}

int64_t cumSumVec(vector<int64_t>& v) {
    int64_t numEls = v.size() - 1;
    int64_t tot = 0;
    for (int64_t i = 0; i < numEls; i++) {
        int64_t oldTot = tot;
        tot += v[i];
        v[i] = oldTot;
    }
    v[numEls] = tot;
    return tot;
}

void rewindVec(std::vector<int64_t>& v, int64_t downTo, int64_t value) {
    for (int64_t i = v.size() - 1; i > downTo; i--) {
        v[i] = v[i - 1];
    }
    v[downTo] = value;
}

}  // end namespace BaSpaCho