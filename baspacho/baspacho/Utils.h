/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace BaSpaCho {

// printable timestamp
std::string timeStamp();

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

using hrc = std::chrono::high_resolution_clock;
using tdelta = std::chrono::duration<double>;

// utility to collect timing stats
template <typename... Args>
struct OpStat {
  struct Instance {
    Instance() : stat(nullptr) {}

    Instance(Instance&& that) : stat(that.stat), start(that.start) { that.stat = nullptr; }

    Instance(OpStat* stat, const Args&... args)
        : stat(stat), start(stat ? hrc::now() : TimePoint()), argsTuple(args...) {}

    template <int i, int j, int... ns>
    inline typename std::enable_if<(i < j)>::type invokeCallback(double runTime) {
      invokeCallback<i + 1, j, ns..., i>(runTime);
    }

    template <int i, int j, int... ns>
    inline typename std::enable_if<(i == j)>::type invokeCallback(double runTime) {
      stat->callBack(runTime, std::get<ns>(argsTuple)...);
    }

    ~Instance() {
      if (!stat) {
        return;
      }
      double runTime = tdelta(hrc::now() - start).count();
      stat->numRuns++;
      stat->lastTime = runTime;
      stat->maxTime = std::max(stat->maxTime, runTime);
      stat->totTime += runTime;
      if (stat->callBack) {
        invokeCallback<0, sizeof...(Args)>(runTime);
      }
    }

    OpStat* stat;
    TimePoint start;
    std::tuple<Args...> argsTuple;
  };

  std::string toString() const {
    std::stringstream ss;
    ss << "#=" << numRuns << ", time=" << totTime << "s, last=" << lastTime << "s, max=" << maxTime
       << "s";
    return ss.str();
  }

  void reset() {
    numRuns = 0;
    totTime = 0;
    maxTime = 0;
    lastTime = 0;
  }

  Instance instance(const Args&... args) { return enabled ? Instance(this, args...) : Instance(); }

  bool enabled = true;
  int64_t numRuns = 0;
  double totTime = 0;
  double maxTime = 0;
  double lastTime = 0;
  std::function<void(double, const Args&... args)> callBack;
};

// utility, check if vector range is strictly increasion
template <typename T>
bool isStrictlyIncreasing(const std::vector<T>& v, std::size_t begin, std::size_t e) {
  std::size_t i = begin + 1;
  while (i < e && (v[i] > v[i - 1])) {
    i++;
  }
  return i == e;
}

// utility, check if vector range is weakly increasion
template <typename T>
bool isWeaklyIncreasing(const std::vector<T>& v, std::size_t begin, std::size_t e) {
  std::size_t i = begin + 1;
  while (i < e && (v[i] >= v[i - 1])) {
    i++;
  }
  return i == e;
}

#ifdef __CUDACC__
#define __BASPACHO_HOST_DEVICE__ __host__ __device__
#else
#define __BASPACHO_HOST_DEVICE__
#endif

template <typename... Args>
__BASPACHO_HOST_DEVICE__ void UNUSED(const Args&... args) {
  (void)(sizeof...(args));
}

__BASPACHO_HOST_DEVICE__
inline int64_t bisect(const int64_t* array, int64_t size, int64_t needle) {
  int64_t a = 0, b = size;
  while (b - a > 1) {
    int64_t m = (a + b) / 2;
    if (needle >= array[m]) {
      a = m;
    } else {
      b = m;
    }
  }
  return a;
}

// appends all elements x shifted by a given amount (x + shift)
template <class ForwardIt>
inline void shiftConcat(std::vector<int64_t>& target, int64_t shift, ForwardIt first,
                        ForwardIt last) {
  while (first != last) {
    target.push_back(shift + *first++);
  }
}

// apply permutation on the left: it[perm[i]] = w[i]
template <typename T, typename ForwardIt>
void leftPermute(ForwardIt it, const std::vector<int64_t>& perm, const std::vector<T>& w) {
  for (size_t i = 0; i < perm.size(); i++) {
    *(it + perm[i]) = w[i];
  }
}

// compute composed permutation v[w[i]]
std::vector<int64_t> composePermutations(const std::vector<int64_t>& v,
                                         const std::vector<int64_t>& w);

// compute inverse permutation
std::vector<int64_t> inversePermutation(const std::vector<int64_t>& v);

// do cumulated sum of v's elements, starting from 0
int64_t cumSumVec(std::vector<int64_t>& v);

// set v[i+1] to v[i] for decreating i, setting v[downTo] = value
void rewindVec(std::vector<int64_t>& v, int64_t downTo = 0, int64_t value = 0);

}  // end namespace BaSpaCho
