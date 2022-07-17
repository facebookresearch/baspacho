
#include "baspacho/baspacho/Utils.h"
#include <ctime>
#include <iomanip>
#include "baspacho/baspacho/DebugMacros.h"

namespace BaSpaCho {

using namespace std;

using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

string timeStamp() {
  using namespace chrono;
  auto now = system_clock::now();
  auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
  const time_t t_c = system_clock::to_time_t(now);
  stringstream ss;
  ss << put_time(localtime(&t_c), "%T") << "." << setfill('0') << setw(3) << ms.count();
  return ss.str();
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