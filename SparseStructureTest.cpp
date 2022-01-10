#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "SparseStructure.h"

using namespace std;
using namespace testing;

TEST(SparseStructure, Transpose) {
    /*
      X _ _ X _
      _ _ X _ X
      X X _ _ X
      _ X X _ _
      _ _ X _ X
    */
    SparseStructure ss({0, 2, 4, 7, 9, 11}, {0, 3, 2, 4, 0, 1, 4, 1, 2, 2, 4});
    SparseStructure t = ss.transpose();

    ASSERT_THAT(t.ptrs, ElementsAre(0, 2, 4, 7, 8, 11));
    ASSERT_THAT(t.inds, ElementsAre(0, 2, 2, 3, 1, 3, 4, 0, 1, 2, 4));
}

TEST(SparseStructure, SymPermutation) {
    /*
      X
      _ X
      X X _
      _ X _ X
      _ _ X _ X
      X X _ _ X X
    */
    SparseStructure ss({0, 1, 2, 4, 6, 8, 12},
                       {0, 1, 0, 1, 1, 3, 2, 4, 0, 1, 4, 5});

    /*
      X
      _ X
      X _ _
      X _ _ X
      _ _ X X X
      _ X X X _ X
    */
    SparseStructure pCsr =
        ss.symmetricPermutation({4, 5, 2, 1, 0, 3}, /* lowerHalf = */ false);

    ASSERT_THAT(pCsr.ptrs, ElementsAre(0, 1, 2, 3, 5, 8, 12));
    ASSERT_THAT(pCsr.inds, ElementsAre(0, 1, 0, 0, 3, 2, 3, 4, 1, 2, 3, 5));

    SparseStructure pCsc =
        ss.symmetricPermutation({4, 5, 2, 1, 0, 3}, /* lowerHalf = */ true);

    ASSERT_THAT(pCsc.ptrs, ElementsAre(0, 3, 5, 7, 10, 11, 12));
    ASSERT_THAT(pCsc.inds, ElementsAre(0, 2, 3, 1, 5, 4, 5, 3, 4, 5, 4, 5));
}

// helper
static std::string printCols(const vector<set<uint64_t>>& columns) {
    stringstream ss;
    ss << "{\n";
    for (const set<uint64_t>& col : columns) {
        ss << "  { ";
        bool first = true;
        for (uint64_t c : col) {
            ss << (first ? "" : ", ") << c;
            first = false;
        }
        ss << " }\n";
    }
    ss << "}";
    return ss.str();
}

static std::string printInts(const vector<uint64_t>& ints) {
    stringstream ss;
    ss << "[";
    bool first = true;
    for (uint64_t c : ints) {
        ss << (first ? "" : ", ") << c;
        first = false;
    }
    ss << "]";
    return ss.str();
}

static vector<set<uint64_t>> randomCols(uint64_t size, double fill,
                                        uint64_t seed) {
    std::mt19937 gen(seed);
    uniform_real_distribution<> dis(0.0, 1.0);
    vector<set<uint64_t>> columns(size);
    for (uint64_t i = 0; i < size; i++) {
        columns[i].insert(i);
        for (uint64_t j = i + 1; j < size; j++) {
            if (dis(gen) < fill) {
                columns[i].insert(j);
            }
        }
    }
    return columns;
}

// helper
static vector<set<uint64_t>> csrStructToColumns(const SparseStructure& mat) {
    uint64_t ord = mat.order();
    vector<set<uint64_t>> columns(ord);
    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = mat.ptrs[i];
        uint64_t end = mat.ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = mat.inds[k];
            columns[j].insert(i);
        }
    }
    return columns;
}

// helper
static SparseStructure columnsToCsrStruct(
    const vector<set<uint64_t>>& columns) {
    vector<uint64_t> ptrs, inds;
    for (const set<uint64_t>& col : columns) {
        ptrs.push_back(inds.size());
        inds.insert(inds.end(), col.begin(), col.end());
    }
    ptrs.push_back(inds.size());
    return SparseStructure(ptrs, inds).transpose();  // csc to csr
}

// naive implementation
static void naiveAddEliminationEntries(vector<set<uint64_t>>& columns,
                                       uint64_t start, uint64_t end) {
    CHECK_LE(end, columns.size());
    for (int i = start; i < end; i++) {
        std::set<uint64_t>& cBlocks = columns[i];
        auto it = cBlocks.begin();
        CHECK(it != cBlocks.end());
        CHECK_EQ(i, *it) << "Expecting diagonal block!";
        while (++it != cBlocks.end()) {
            auto it2 = it;
            std::set<uint64_t>& cAltBlocks = columns[*it];
            while (++it2 != cBlocks.end()) {
                cAltBlocks.insert(*it2);
            }
        }
    }
}

TEST(SparseStructure, EliminationEntries) {
    vector<uint64_t> sizes{10, 20, 30, 40};
    vector<double> fills{0.15, 0.23, 0.3};
    int seed = 37;
    for (auto size : sizes) {
        for (auto fill : fills) {
            auto colsOrig = randomCols(size, fill, seed++);
            auto ssOrig = columnsToCsrStruct(colsOrig);

            LOG(INFO) << "ptrs: " << printInts(ssOrig.ptrs);
            LOG(INFO) << "inds: " << printInts(ssOrig.inds);
            LOG(INFO) << "orig: "
                      << printCols(csrStructToColumns(ssOrig.transpose()));

            for (int start = 0; start < size * 2 / 3; start += 3) {
                for (int end = start + 3; end < size; end += 3) {
                    vector<set<uint64_t>> cols = colsOrig;
                    naiveAddEliminationEntries(cols, start, end);
                    auto ssElim = columnsToCsrStruct(cols);

                    LOG(INFO) << "gt.ptrs: " << printInts(ssElim.ptrs);
                    // LOG(INFO) << "gt.inds: " << printInts(ssElim.inds);
                    LOG(INFO)
                        << "gt: "
                        << printCols(csrStructToColumns(ssElim.transpose()));

                    auto ssElim2 = ssOrig.addEliminationEntries(start, end);

                    LOG(INFO) << "xx.ptrs: " << printInts(ssElim2.ptrs);
                    // LOG(INFO) << "xx.inds: " << printInts(ssElim2.inds);
                    LOG(INFO)
                        << "xx: "
                        << printCols(csrStructToColumns(ssElim2.transpose()));

                    LOG(INFO) << "start: " << start << ", end: " << end;
                    LOG(INFO) << "ptrs after:\n" << printInts(ssElim.ptrs);
                    LOG(INFO) << "ptrs exper:\n" << printInts(ssElim2.ptrs);

                    ASSERT_THAT(ssElim.ptrs,
                                testing::ContainerEq(ssElim2.ptrs));
                    // LOG(INFO) << "ptrs exper:\n" << printInts(ssElim2.ptrs);
                    // LOG(INFO) << "inds exper:\n" << printInts(ssElim2.inds);
                }
            }
        }
    }
}

/*TEST(SparseStructure, EliminationEntries) {
    auto cols = randomCols(10, 0.3, 37);
    LOG(INFO) << "before: " << printCols(cols);
    auto ss = columnsToCsrStruct(cols);

    naiveAddEliminationEntries(cols, 3, 10);
    LOG(INFO) << "after: " << printCols(cols);
    auto ssElim = columnsToCsrStruct(cols);

    LOG(INFO) << "ptrs before:\n" << printInts(ss.ptrs);
    LOG(INFO) << "ptrs after:\n" << printInts(ssElim.ptrs);
    LOG(INFO) << "inds after:\n" << printInts(ssElim.inds);

    auto ssElim2 = ss.addEliminationEntries(3, 10);
    LOG(INFO) << "ptrs exper:\n" << printInts(ssElim2.ptrs);
    LOG(INFO) << "inds exper:\n" << printInts(ssElim2.inds);
}*/