/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/baspacho/Utils.h"
#include <cmath>
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
  struct tm local_tm;
  ss << put_time(localtime_r(&t_c, &local_tm), "%T") << "." << setfill('0') << setw(3)
     << ms.count();
  return ss.str();
}

void throwError(const char* file, int line, const char* msg) {
  std::stringstream s;
  s << "[" << timeStamp() << " " << file << ":" << line << "] Check failed: " << msg;
  throw std::runtime_error(s.str());
}

string microsecondsString(size_t microseconds, int precision) {
  ostringstream os;
  if (microseconds < 1000) {
    os << microseconds << "\u03bcs";
  } else if (microseconds < 100000) {
    os << fixed << setprecision(precision) << (microseconds * 0.001) << "ms";
  } else {
    if (microseconds >= 3600000000L) {
      os << microseconds / 3600000000 << "h";
      microseconds %= 3600000000;
      if (microseconds >= 60000000) {
        os << microseconds / 60000000 << "m";
      }
    } else if (microseconds >= 60000000) {
      os << microseconds / 60000000 << "m";
      microseconds %= 60000000;
      size_t seconds = (size_t)round(microseconds * 0.000001);
      if (seconds > 0) {
        os << seconds << "s";
      }
    } else {
      os << fixed << setprecision(precision) << (microseconds * 0.000001) << "s";
    }
  }
  return os.str();
}

std::string secondsToString(double secs, int precision) {
  return microsecondsString((size_t)std::round(secs * 1000000), precision);
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
