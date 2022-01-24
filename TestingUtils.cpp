
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

SparseMatGenerator::SparseMatGenerator(uint64_t size, int64_t seed)
    : gen(seed), columns(size) {
    for (uint64_t i = 0; i < size; i++) {
        columns[i].insert(i);
    }
}

void SparseMatGenerator::connectRanges(int64_t begin1, int64_t end1,
                                       int64_t begin2, int64_t end2,
                                       double fill, int64_t maxOffset) {
    if (begin1 > begin2) {
        connectRanges(begin2, end2, begin1, end1, fill, maxOffset);
        return;
    }

    if (end1 > end2) {
        connectRanges(begin2, end2, end2, end1, fill, maxOffset);
    }

    uniform_real_distribution<> dis(0.0, 1.0);
    for (int64_t i = begin1; i < end1; i++) {
        int64_t dBegin = min(maxOffset, max(begin2 - i, 1L));
        int64_t dEnd = min(maxOffset, end2 - i);
        for (int64_t j = i + dBegin; j < i + dEnd; j++) {
            if (fill >= 1.0 || fill > dis(gen)) {
                columns[i].insert(j);
            }
        }
    }
}

SparseMatGenerator SparseMatGenerator::genFlat(int64_t size, double fill,
                                               int64_t seed) {
    SparseMatGenerator retv(size, seed);
    retv.connectRanges(0, size, 0, size, fill);
    return retv;
}

// topology is roughly a line, entries in band are set with a probability
SparseMatGenerator SparseMatGenerator::genLine(int64_t size, double fill,
                                               int64_t bandSize, int64_t seed) {
    SparseMatGenerator retv(size, seed);
    retv.connectRanges(0, size, 0, size, fill);
    return retv;
}

// topology is a set of meridians (connecting north and south poles)
SparseMatGenerator SparseMatGenerator::genMeridians(
    int64_t num, int64_t lineLen, double fill, int64_t bandSize,
    int64_t hairLen, int64_t nPoleHairs, int64_t sPoleHairs, int64_t seed) {
    int64_t totHairs = nPoleHairs + sPoleHairs;
    int64_t size = lineLen * num + hairLen * totHairs;
    int64_t endMeridians = lineLen * num;
    SparseMatGenerator retv(size, seed);
    // build structure of meridians and hairs
    for (int64_t i = 0; i < num; i++) {
        retv.connectRanges(lineLen * i, lineLen, lineLen * i, lineLen, fill,
                           bandSize);
    }
    for (int64_t i = 0; i < totHairs; i++) {
        retv.connectRanges(endMeridians + hairLen * i, hairLen,
                           endMeridians + hairLen * i, hairLen, fill, bandSize);
    }
    // connect meridians meeting at pole
    for (int64_t i = 0; i < num; i++) {
        for (uint64_t j = 0; j < i; j++) {
        }
    }
    return retv;
}