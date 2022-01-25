
#include "TestingMatGen.h"

#include <glog/logging.h>

using namespace std;

SparseMatGenerator::SparseMatGenerator(int64_t size, int64_t seed)
    : gen(seed), columns(size) {
    for (int64_t i = 0; i < size; i++) {
        columns[i].insert(i);
    }
}

void SparseMatGenerator::connectRanges(int64_t begin1, int64_t end1,
                                       int64_t begin2, int64_t end2,
                                       double fill, int64_t maxOffset) {
    CHECK_GE(begin1, 0);
    CHECK_GE(begin2, 0);
    CHECK_LE(end1, columns.size());
    CHECK_LE(end2, columns.size());

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

void SparseMatGenerator::addSparseConnections(double fill) {
    connectRanges(0, columns.size(), 0, columns.size(), fill);
}

void SparseMatGenerator::addSchurSet(int64_t size, double fill) {
    vector<set<uint64_t>> newCols(size + columns.size());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int64_t i = 0; i < size; i++) {
        newCols[i].insert(i);
        for (int64_t j = size; j < (int64_t)newCols.size(); j++) {
            if (fill >= 1.0 || fill > dis(gen)) {
                newCols[i].insert(j);
            }
        }
    }
    // shift
    for (uint64_t i = 0; i < columns.size(); i++) {
        for (uint64_t j : columns[i]) {
            newCols[i + size].insert(j + size);
        }
    }
    swap(newCols, columns);
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

    CHECK_LE(bandSize, lineLen);
    CHECK_LE(bandSize, hairLen);

    SparseMatGenerator retv(size, seed);
    // build structure of meridians and hairs
    for (int64_t i = 0; i < num; i++) {
        int64_t iBegin = lineLen * i;
        retv.connectRanges(iBegin, iBegin + lineLen, iBegin, iBegin + lineLen,
                           fill, bandSize);
    }
    for (int64_t h = 0; h < totHairs; h++) {
        int64_t hBegin = endMeridians + hairLen * h;
        retv.connectRanges(hBegin, hBegin + hairLen, hBegin, hBegin + hairLen,
                           fill, bandSize);
    }
    // connect meridians meeting at pole (begin = north, end = south)
    for (int64_t i = 0; i < num; i++) {
        int64_t iBegin = lineLen * i;
        for (uint64_t j = 0; j < i; j++) {
            int64_t jBegin = lineLen * j;
            // conenct north pole ends
            retv.connectRanges(iBegin, iBegin + bandSize,  //
                               jBegin, jBegin + bandSize,  //
                               fill, bandSize);
            // connect south pole ends
            retv.connectRanges(
                iBegin + lineLen - bandSize, iBegin + lineLen,  //
                jBegin + lineLen - bandSize, jBegin + lineLen,  //
                fill, bandSize);
        }
    }

    // connect hairs to meridians
    for (int64_t i = 0; i < num; i++) {
        int64_t iBegin = lineLen * i;

        // north pole ends
        for (int64_t h = 0; h < nPoleHairs; h++) {
            int64_t hBegin = endMeridians + hairLen * h;
            retv.connectRanges(iBegin, iBegin + bandSize,  //
                               hBegin, hBegin + bandSize,  //
                               fill, bandSize);
        }

        // south pole ends
        for (int64_t h = 0; h < sPoleHairs; h++) {
            int64_t hBegin = endMeridians + hairLen * (h + nPoleHairs);
            retv.connectRanges(iBegin + lineLen - bandSize,
                               iBegin + lineLen,           //
                               hBegin, hBegin + bandSize,  //
                               fill, bandSize);
        }
    }

    // conenct hair to hair at poles
    for (int64_t h = 0; h < nPoleHairs; h++) {
        int64_t hBegin = endMeridians + hairLen * h;
        for (int64_t k = 0; k < h; k++) {
            int64_t kBegin = endMeridians + hairLen * k;
            retv.connectRanges(kBegin, kBegin + bandSize,  //
                               hBegin, hBegin + bandSize,  //
                               fill, bandSize);
        }
    }

    for (int64_t h = 0; h < sPoleHairs; h++) {
        int64_t hBegin = endMeridians + hairLen * (h + nPoleHairs);
        for (int64_t k = 0; k < h; k++) {
            int64_t kBegin = endMeridians + hairLen * (h + nPoleHairs);
            retv.connectRanges(kBegin, kBegin + bandSize,  //
                               hBegin, hBegin + bandSize,  //
                               fill, bandSize);
        }
    }

    return retv;
}

SparseMatGenerator SparseMatGenerator::genGrid(int64_t width, int64_t height,
                                               double fill, int64_t connMaxDist,
                                               int64_t seed) {
    int64_t sz = width * height;
    SparseMatGenerator retv(sz, seed);

    uniform_real_distribution<> dis(0.0, 1.0);
    for (int64_t i = 0; i < width; i++) {
        int64_t i2begin = max(i - connMaxDist, 0L);
        int64_t i2end = min(i + connMaxDist + 1, width);

        for (int64_t j = 0; j < height; j++) {
            int64_t j2begin = max(j - connMaxDist, 0L);
            int64_t j2end = min(j + connMaxDist + 1, height);
            int64_t off = i * height + j;
            for (int i2 = i2begin; i2 < i2end; i2++) {
                for (int j2 = j2begin; j2 < j2end; j2++) {
                    if (i2 == i && j2 == j) {
                        continue;
                    }
                    if (fill >= 1.0 || fill > dis(retv.gen)) {
                        int64_t off2 = i2 * height + j2;
                        int64_t offMin = min(off, off2);
                        int64_t offMax = max(off, off2);
                        retv.columns[offMin].insert(offMax);
                    }
                }
            }
        }
    }

    return retv;
}