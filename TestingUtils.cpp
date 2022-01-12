
#include "TestingUtils.h"

#include <glog/logging.h>

#include <random>

using namespace std;

string printCols(const vector<set<uint64_t>>& columns) {
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

vector<set<uint64_t>> randomCols(uint64_t size, double fill, uint64_t seed) {
    mt19937 gen(seed);
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
vector<set<uint64_t>> csrStructToColumns(const SparseStructure& mat) {
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
SparseStructure columnsToCsrStruct(const vector<set<uint64_t>>& columns) {
    vector<uint64_t> ptrs, inds;
    for (const set<uint64_t>& col : columns) {
        ptrs.push_back(inds.size());
        inds.insert(inds.end(), col.begin(), col.end());
    }
    ptrs.push_back(inds.size());
    return SparseStructure(ptrs, inds).transpose();  // csc to csr
}

// naive implementation
void naiveAddEliminationEntries(vector<set<uint64_t>>& columns, uint64_t start,
                                uint64_t end) {
    CHECK_LE(end, columns.size());
    for (int i = start; i < end; i++) {
        set<uint64_t>& cBlocks = columns[i];
        auto it = cBlocks.begin();
        CHECK(it != cBlocks.end());
        CHECK_EQ(i, *it) << "Expecting diagonal block!";
        while (++it != cBlocks.end()) {
            auto it2 = it;
            set<uint64_t>& cAltBlocks = columns[*it];
            while (++it2 != cBlocks.end()) {
                cAltBlocks.insert(*it2);
            }
        }
    }
}

vector<set<uint64_t>> makeIndependentElimSet(vector<set<uint64_t>>& columns,
                                             uint64_t start, uint64_t end) {
    vector<set<uint64_t>> retvCols(columns.size());
    for (size_t i = 0; i < columns.size(); i++) {
        if (i < start || i >= end) {
            retvCols[i] = columns[i];
        } else {
            retvCols[i].insert(i);
            for (uint64_t c : columns[i]) {
                if (c >= end) {
                    retvCols[i].insert(c);
                }
            }
        }
    }
    return retvCols;
}