#pragma once

#include <vector>

template <typename T>
bool isStrictlyIncreasing(const std::vector<T>& v, size_t begin, size_t e) {
    size_t i = begin + 1;
    while (i < e && (v[i] > v[i - 1])) {
        i++;
    }
    return i == e;
}

template <typename T>
bool isWeaklyIncreasing(const std::vector<T>& v, size_t begin, size_t e) {
    size_t i = begin + 1;
    while (i < e && (v[i] >= v[i - 1])) {
        i++;
    }
    return i == e;
}
