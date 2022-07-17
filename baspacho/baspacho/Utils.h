#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace BaSpaCho {

std::string timeStamp();

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

using hrc = std::chrono::high_resolution_clock;
using tdelta = std::chrono::duration<double>;

/*struct OpStat {
  int64_t numRuns = 0;
  double totTime = 0;
  double maxTime = 0;
  double lastTime = 0;
  std::string toString() const;
  void reset();
};

struct OpInstance {
  OpInstance(OpStat& stat);
  ~OpInstance();
  OpStat& stat;
  TimePoint start;
};*/

template <typename... Args>
struct OpStat {
  struct Instance {
    Instance() : stat(nullptr) {}

    Instance(Instance&& that) : stat(that.stat), start(that.start) { that.stat = nullptr; }

    Instance(OpStat* stat, const Args&... args)
        : stat(stat), start(stat ? hrc::now() : TimePoint()), argsTuple(args...) {}

    ~Instance() {
      if (!stat) {
        return;
      }
      double time = tdelta(hrc::now() - start).count();
      stat->numRuns++;
      stat->lastTime = time;
      stat->maxTime = std::max(stat->maxTime, time);
      stat->totTime += time;
      if (stat->callBack) {
        std::apply([&](auto&&... args) { stat->callBack(time, args...); }, argsTuple);
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

template <typename T>
bool isStrictlyIncreasing(const std::vector<T>& v, std::size_t begin, std::size_t e) {
  std::size_t i = begin + 1;
  while (i < e && (v[i] > v[i - 1])) {
    i++;
  }
  return i == e;
}

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

template <typename T, typename ForwardIt>
void leftPermute(ForwardIt it, const std::vector<int64_t>& perm, const std::vector<T>& w) {
  for (size_t i = 0; i < perm.size(); i++) {
    *(it + perm[i]) = w[i];
  }
}

std::vector<int64_t> composePermutations(const std::vector<int64_t>& v,
                                         const std::vector<int64_t>& w);

std::vector<int64_t> inversePermutation(const std::vector<int64_t>& v);

int64_t cumSumVec(std::vector<int64_t>& v);

void rewindVec(std::vector<int64_t>& v, int64_t downTo = 0, int64_t value = 0);

}  // end namespace BaSpaCho