#pragma once

#include <set>
#include <sstream>
#include <string>

#include "SparseStructure.h"

template <typename T>
std::string printInts(const std::vector<T>& ints) {
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

std::vector<std::set<uint64_t>> randomCols(uint64_t size, double fill,
                                           uint64_t seed);

std::vector<std::set<uint64_t>> csrStructToColumns(const SparseStructure& mat);

SparseStructure columnsToCsrStruct(
    const std::vector<std::set<uint64_t>>& columns);

void naiveAddEliminationEntries(std::vector<std::set<uint64_t>>& columns,
                                uint64_t start, uint64_t end);

std::vector<std::set<uint64_t>> makeIndependentElimSet(
    std::vector<std::set<uint64_t>>& columns, uint64_t start, uint64_t end);