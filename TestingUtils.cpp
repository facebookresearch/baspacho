
#include "TestingUtils.h"

#include <glog/logging.h>

#include <algorithm>
#include <random>

using namespace std;

vector<uint64_t> randomPermutation(size_t size, uint64_t seed) {
    mt19937 gen(seed);
    vector<uint64_t> retv(size);
    iota(retv.begin(), retv.end(), 0);
    shuffle(retv.begin(), retv.end(), gen);
    return retv;
}

vector<uint64_t> randomVec(size_t size, uint64_t low, uint64_t high,
                           uint64_t seed) {
    mt19937 gen(seed);
    vector<uint64_t> retv(size);
    uniform_int_distribution el(low, high);
    for (uint64_t i = 0; i < size; i++) {
        retv[i] = el(gen);
    }
    return retv;
}

vector<uint64_t> randomPartition(size_t weight, uint64_t low, uint64_t high,
                                 uint64_t seed) {
    mt19937 gen(seed);
    vector<uint64_t> retv;
    uniform_int_distribution<uint64_t> el(low, high);
    while (weight > 0) {
        uint64_t val = std::min(weight, el(gen));
        retv.push_back(val);
        weight -= val;
    }
    return retv;
}

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

string printPattern(const SparseStructure& mat, bool sym) {
    uint64_t ord = mat.order();
    vector<bool> isSet(ord * ord, false);
    for (uint64_t i = 0; i < ord; i++) {
        uint64_t start = mat.ptrs[i];
        uint64_t end = mat.ptrs[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = mat.inds[k];
            isSet[ord * i + j] = true;  // assume CSR
            if (sym) {
                isSet[ord * j + i] = true;
            }
        }
    }
    stringstream ss;
    for (uint64_t i = 0; i < ord; i++) {
        for (uint64_t j = 0; j < ord; j++) {
            ss << (j > 0 ? " " : "") << (isSet[ord * i + j] ? "#" : "_");
        }
        ss << "\n";
    }
    return ss.str();
}

// print sparse structure after collapsing columns according to lumpStart
string printAggreg(vector<uint64_t> ptrs,  // csc
                   vector<uint64_t> inds, vector<uint64_t> lumpStart) {
    CHECK_EQ(ptrs.size(), lumpStart.size());
    int64_t ord = lumpStart[lumpStart.size() - 1];
    vector<bool> isSet(ord * ord, false);
    for (uint64_t i = 0; i < ptrs.size() - 1; i++) {
        uint64_t start = ptrs[i];
        uint64_t end = ptrs[i + 1];
        uint64_t aStart = lumpStart[i];
        uint64_t aEnd = lumpStart[i + 1];
        for (uint64_t k = start; k < end; k++) {
            uint64_t j = inds[k];
            for (uint64_t a = aStart; a < aEnd; a++) {
                isSet[ord * j + a] = true;  // assume CSR
            }
        }
    }
    stringstream ss;
    for (uint64_t i = 0; i < ord; i++) {
        for (uint64_t j = 0; j < ord; j++) {
            ss << (j > 0 ? " " : "") << (isSet[ord * i + j] ? "#" : "_");
        }
        ss << "\n";
    }
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

std::vector<std::set<uint64_t>> joinColums(
    const std::vector<std::set<uint64_t>>& columns,
    std::vector<uint64_t> lumpStart) {
    CHECK_EQ(lumpStart[lumpStart.size() - 1], columns.size());
    std::vector<std::set<uint64_t>> retv;
    for (uint64_t a = 0; a < lumpStart.size() - 1; a++) {
        uint64_t start = lumpStart[a];
        uint64_t end = lumpStart[a + 1];
        std::set<uint64_t> colz;
        for (uint64_t i = start; i < end; i++) {
            colz.insert(columns[i].begin(), columns[i].end());
        }
        retv.push_back(colz);
    }
    return retv;
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
SparseStructure columnsToCscStruct(const vector<set<uint64_t>>& columns) {
    vector<uint64_t> ptrs, inds;
    for (const set<uint64_t>& col : columns) {
        ptrs.push_back(inds.size());
        inds.insert(inds.end(), col.begin(), col.end());
    }
    ptrs.push_back(inds.size());
    return SparseStructure(ptrs, inds);  // csc to csr
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
