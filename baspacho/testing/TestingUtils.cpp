/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "baspacho/testing/TestingUtils.h"
#include <algorithm>
#include <random>
#include "baspacho/baspacho/DebugMacros.h"

namespace BaSpaCho::testing_utils {
using namespace std;

vector<int64_t> randomPermutation(size_t size, int64_t seed) {
  mt19937 gen(seed);
  vector<int64_t> retv(size);
  iota(retv.begin(), retv.end(), 0);
  shuffle(retv.begin(), retv.end(), gen);
  return retv;
}

vector<int64_t> randomVec(size_t size, int64_t low, int64_t high, int64_t seed) {
  mt19937 gen(seed);
  return randomVec(size, low, high, gen);
}

vector<int64_t> randomVec(size_t size, int64_t low, int64_t high, std::mt19937& gen) {
  vector<int64_t> retv(size);
  uniform_int_distribution el(low, high);
  for (int64_t i = 0; i < size; i++) {
    retv[i] = el(gen);
  }
  return retv;
}

template <typename T>
std::vector<T> randomData(size_t size, T low, T high, int64_t seed) {
  mt19937 gen(seed);
  return randomData(size, low, high, gen);
}

template <typename T>
std::vector<T> randomData(size_t size, T low, T high, std::mt19937& gen) {
  vector<T> retv(size);
  uniform_real_distribution<> dis(low, high);
  for (int64_t i = 0; i < size; i++) {
    retv[i] = dis(gen);
  }
  return retv;
}

template std::vector<double> randomData(size_t size, double low, double high, int64_t seed);
template std::vector<float> randomData(size_t size, float low, float high, int64_t seed);
template std::vector<double> randomData(size_t size, double low, double high, std::mt19937& gen);
template std::vector<float> randomData(size_t size, float low, float high, std::mt19937& gen);

vector<int64_t> randomPartition(int64_t weight, int64_t low, int64_t high, int64_t seed) {
  mt19937 gen(seed);
  vector<int64_t> retv;
  uniform_int_distribution<int64_t> el(low, high);
  while (weight > 0) {
    int64_t val = std::min(weight, el(gen));
    retv.push_back(val);
    weight -= val;
  }
  return retv;
}

string printCols(const vector<set<int64_t>>& columns) {
  stringstream ss;
  ss << "{\n";
  for (const set<int64_t>& col : columns) {
    ss << "  { ";
    bool first = true;
    for (int64_t c : col) {
      ss << (first ? "" : ", ") << c;
      first = false;
    }
    ss << " }\n";
  }
  ss << "}";
  return ss.str();
}

string printPattern(const SparseStructure& mat, bool sym) {
  int64_t ord = mat.order();
  vector<bool> isSet(ord * ord, false);
  for (int64_t i = 0; i < ord; i++) {
    int64_t start = mat.ptrs[i];
    int64_t end = mat.ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = mat.inds[k];
      isSet[ord * i + j] = true;  // assume CSR
      if (sym) {
        isSet[ord * j + i] = true;
      }
    }
  }
  stringstream ss;
  for (int64_t i = 0; i < ord; i++) {
    for (int64_t j = 0; j < ord; j++) {
      ss << (j > 0 ? " " : "") << (isSet[ord * i + j] ? "#" : "_");
    }
    ss << "\n";
  }
  return ss.str();
}

// print sparse structure after collapsing columns according to lumpStart
string printAggreg(vector<int64_t> ptrs,  // csc
                   vector<int64_t> inds, vector<int64_t> lumpStart) {
  BASPACHO_CHECK_EQ(ptrs.size(), lumpStart.size());
  int64_t ord = lumpStart[lumpStart.size() - 1];
  vector<bool> isSet(ord * ord, false);
  for (int64_t i = 0; i < ptrs.size() - 1; i++) {
    int64_t start = ptrs[i];
    int64_t end = ptrs[i + 1];
    int64_t aStart = lumpStart[i];
    int64_t aEnd = lumpStart[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = inds[k];
      for (int64_t a = aStart; a < aEnd; a++) {
        isSet[ord * j + a] = true;  // assume CSR
      }
    }
  }
  stringstream ss;
  for (int64_t i = 0; i < ord; i++) {
    for (int64_t j = 0; j < ord; j++) {
      ss << (j > 0 ? " " : "") << (isSet[ord * i + j] ? "#" : "_");
    }
    ss << "\n";
  }
  return ss.str();
}

vector<set<int64_t>> randomCols(int64_t size, double fill, int64_t seed) {
  mt19937 gen(seed);
  uniform_real_distribution<> dis(0.0, 1.0);
  vector<set<int64_t>> columns(size);
  for (int64_t i = 0; i < size; i++) {
    columns[i].insert(i);
    for (int64_t j = i + 1; j < size; j++) {
      if (dis(gen) < fill) {
        columns[i].insert(j);
      }
    }
  }
  return columns;
}

std::vector<std::set<int64_t>> joinColums(const std::vector<std::set<int64_t>>& columns,
                                          std::vector<int64_t> lumpStart) {
  BASPACHO_CHECK_EQ(lumpStart[lumpStart.size() - 1], columns.size());
  std::vector<std::set<int64_t>> retv;
  for (int64_t a = 0; a < lumpStart.size() - 1; a++) {
    int64_t start = lumpStart[a];
    int64_t end = lumpStart[a + 1];
    std::set<int64_t> colz;
    for (int64_t i = start; i < end; i++) {
      colz.insert(columns[i].begin(), columns[i].end());
    }
    retv.push_back(colz);
  }
  return retv;
}

// helper
vector<set<int64_t>> csrStructToColumns(const SparseStructure& mat) {
  int64_t ord = mat.order();
  vector<set<int64_t>> columns(ord);
  for (int64_t i = 0; i < ord; i++) {
    int64_t start = mat.ptrs[i];
    int64_t end = mat.ptrs[i + 1];
    for (int64_t k = start; k < end; k++) {
      int64_t j = mat.inds[k];
      columns[j].insert(i);
    }
  }
  return columns;
}

// helper
SparseStructure columnsToCscStruct(const vector<set<int64_t>>& columns) {
  vector<int64_t> ptrs, inds;
  for (const set<int64_t>& col : columns) {
    ptrs.push_back(inds.size());
    inds.insert(inds.end(), col.begin(), col.end());
  }
  ptrs.push_back(inds.size());
  return SparseStructure(ptrs, inds);  // csc to csr
}

// naive implementation
void naiveAddEliminationEntries(vector<set<int64_t>>& columns, int64_t start, int64_t end) {
  BASPACHO_CHECK_LE(end, columns.size());
  for (int i = start; i < end; i++) {
    set<int64_t>& cBlocks = columns[i];
    auto it = cBlocks.begin();
    BASPACHO_CHECK(it != cBlocks.end());
    BASPACHO_CHECK_EQ(i, *it);  // Expecting diagonal block!;
    while (++it != cBlocks.end()) {
      auto it2 = it;
      set<int64_t>& cAltBlocks = columns[*it];
      while (++it2 != cBlocks.end()) {
        cAltBlocks.insert(*it2);
      }
    }
  }
}

vector<set<int64_t>> makeIndependentElimSet(vector<set<int64_t>>& columns, int64_t start,
                                            int64_t end) {
  vector<set<int64_t>> retvCols(columns.size());
  for (size_t i = 0; i < columns.size(); i++) {
    if (i < start || i >= end) {
      retvCols[i] = columns[i];
    } else {
      retvCols[i].insert(i);
      for (int64_t c : columns[i]) {
        if (c >= end) {
          retvCols[i].insert(c);
        }
      }
    }
  }
  return retvCols;
}

}  // end namespace BaSpaCho::testing_utils
