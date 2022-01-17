#pragma once

#include <set>
#include <sstream>
#include <string>

#include "SparseStructure.h"

std::vector<uint64_t> randomPermutation(size_t size, uint64_t seed);

std::vector<uint64_t> randomVec(size_t size, uint64_t low, uint64_t high,
                                uint64_t seed);

std::vector<uint64_t> randomPartition(size_t weight, uint64_t low,
                                      uint64_t high, uint64_t seed);

template <typename T>
std::string printVec(const std::vector<T>& ints) {
    std::stringstream ss;
    ss << "[";
    bool first = true;
    for (auto c : ints) {
        ss << (first ? "" : ", ") << c;
        first = false;
    }
    ss << "]";
    return ss.str();
}

std::string printCols(const std::vector<std::set<uint64_t>>& columns);

std::string printPattern(const SparseStructure& mat, bool sym);

std::string printAggreg(std::vector<uint64_t> ptrs, std::vector<uint64_t> inds,
                        std::vector<uint64_t> aggregStart);

std::vector<std::set<uint64_t>> randomCols(uint64_t size, double fill,
                                           uint64_t seed);

std::vector<std::set<uint64_t>> joinColums(
    const std::vector<std::set<uint64_t>>& columns,
    std::vector<uint64_t> aggregStart);

std::vector<std::set<uint64_t>> csrStructToColumns(const SparseStructure& mat);

SparseStructure columnsToCscStruct(
    const std::vector<std::set<uint64_t>>& columns);

void naiveAddEliminationEntries(std::vector<std::set<uint64_t>>& columns,
                                uint64_t start, uint64_t end);

std::vector<std::set<uint64_t>> makeIndependentElimSet(
    std::vector<std::set<uint64_t>>& columns, uint64_t start, uint64_t end);