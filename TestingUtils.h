#pragma once

#include <limits>
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
                        std::vector<uint64_t> lumpStart);

std::vector<std::set<uint64_t>> randomCols(uint64_t size, double fill,
                                           uint64_t seed);

std::vector<std::set<uint64_t>> joinColums(
    const std::vector<std::set<uint64_t>>& columns,
    std::vector<uint64_t> lumpStart);

std::vector<std::set<uint64_t>> csrStructToColumns(const SparseStructure& mat);

SparseStructure columnsToCscStruct(
    const std::vector<std::set<uint64_t>>& columns);

void naiveAddEliminationEntries(std::vector<std::set<uint64_t>>& columns,
                                uint64_t start, uint64_t end);

std::vector<std::set<uint64_t>> makeIndependentElimSet(
    std::vector<std::set<uint64_t>>& columns, uint64_t start, uint64_t end);

//
class SparseMatGenerator {
    SparseMatGenerator(int64_t size, int64_t seed = 37);

    void connectRanges(int64_t begin1, int64_t end1, int64_t begin2,
                       int64_t end2, double fill,
                       int64_t maxOffset = std::numeric_limits<int64_t>::max());

    static SparseMatGenerator genFlat(int64_t size, double fill,
                                      int64_t seed = 37);

    // topology is roughly a line, entries in band are set with a probability
    static SparseMatGenerator genLine(int64_t size, double fill,
                                      int64_t bandSize, int64_t seed = 37);

    // topology is a set of meridians (connecting north and south poles)
    static SparseMatGenerator genMeridians(int64_t num, int64_t lineLen,
                                           double fill, int64_t bandSize,
                                           int64_t nPoleHairs,
                                           int64_t sPoleHairs,
                                           int64_t seed = 37);

    mt19937 gen;
    std::vector<std::set<uint64_t>> columns;
};