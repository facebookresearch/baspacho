#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct OpStat {
    uint64_t numRuns = 0;
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

std::vector<uint64_t> composePermutations(const std::vector<uint64_t>& v,
                                          const std::vector<uint64_t>& w);

std::vector<uint64_t> inversePermutation(const std::vector<uint64_t>& v);

uint64_t cumSumVec(std::vector<uint64_t>& v);

void rewindVec(std::vector<uint64_t>& v, uint64_t downTo = 0,
               uint64_t value = 0);