/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include "baspacho/baspacho/SparseStructure.h"

namespace BaSpaCho::testing_utils {

std::vector<int64_t> randomPermutation(size_t size, int64_t seed);

std::vector<int64_t> randomVec(size_t size, int64_t low, int64_t high, int64_t seed);

std::vector<int64_t> randomVec(size_t size, int64_t low, int64_t high, std::mt19937& gen);

template <typename T>
std::vector<T> randomData(size_t size, T low, T high, int64_t seed);

template <typename T>
std::vector<T> randomData(size_t size, T low, T high, std::mt19937& gen);

std::vector<int64_t> randomPartition(int64_t weight, int64_t low, int64_t high, int64_t seed);

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

std::string printCols(const std::vector<std::set<int64_t>>& columns);

std::string printPattern(const SparseStructure& mat, bool sym);

std::string printAggreg(std::vector<int64_t> ptrs, std::vector<int64_t> inds,
                        std::vector<int64_t> lumpStart);

std::vector<std::set<int64_t>> randomCols(int64_t size, double fill, int64_t seed);

std::vector<std::set<int64_t>> joinColums(const std::vector<std::set<int64_t>>& columns,
                                          std::vector<int64_t> lumpStart);

std::vector<std::set<int64_t>> csrStructToColumns(const SparseStructure& mat);

SparseStructure columnsToCscStruct(const std::vector<std::set<int64_t>>& columns);

void naiveAddEliminationEntries(std::vector<std::set<int64_t>>& columns, int64_t start,
                                int64_t end);

std::vector<std::set<int64_t>> makeIndependentElimSet(std::vector<std::set<int64_t>>& columns,
                                                      int64_t start, int64_t end);

}  // end namespace BaSpaCho::testing_utils
