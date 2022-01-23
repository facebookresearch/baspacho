#pragma once

#include <cstdint>
#include <vector>

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