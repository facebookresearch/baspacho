#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace BaSpaCho {

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct OpStat {
    int64_t numRuns = 0;
    double totTime = 0;
    double maxTime = 0;
    double lastTime = 0;
    std::string toString() const;
};

struct OpInstance {
    OpInstance(OpStat& stat);
    ~OpInstance();
    OpStat& stat;
    TimePoint start;
};

template <typename T>
bool isStrictlyIncreasing(const std::vector<T>& v, std::size_t begin,
                          std::size_t e) {
    std::size_t i = begin + 1;
    while (i < e && (v[i] > v[i - 1])) {
        i++;
    }
    return i == e;
}

template <typename T>
bool isWeaklyIncreasing(const std::vector<T>& v, std::size_t begin,
                        std::size_t e) {
    std::size_t i = begin + 1;
    while (i < e && (v[i] >= v[i - 1])) {
        i++;
    }
    return i == e;
}

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

std::vector<int64_t> composePermutations(const std::vector<int64_t>& v,
                                         const std::vector<int64_t>& w);

std::vector<int64_t> inversePermutation(const std::vector<int64_t>& v);

int64_t cumSumVec(std::vector<int64_t>& v);

void rewindVec(std::vector<int64_t>& v, int64_t downTo = 0, int64_t value = 0);

}  // end namespace BaSpaCho