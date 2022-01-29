
#include <Eigen/Geometry>
#include <tuple>

#include "Utils.h"

struct CoalescedAccessor {
    CoalescedAccessor(const uint64_t* spanStart, const uint64_t* spanToLump,
                      const uint64_t* lumpStart,
                      const uint64_t* spanOffsetInLump,
                      const uint64_t* chainColPtr, const uint64_t* chainRowSpan,
                      const uint64_t* chainData)
        : spanStart(spanStart),
          spanToLump(spanToLump),
          lumpStart(lumpStart),
          spanOffsetInLump(spanOffsetInLump),
          chainColPtr(chainColPtr),
          chainRowSpan(chainRowSpan),
          chainData(chainData) {}

    uint64_t paramSize(uint64_t blockIndex) const {
        return spanStart[blockIndex + 1] - spanStart[blockIndex];
    }

    uint64_t paramStart(uint64_t blockIndex) const {
        return spanStart[blockIndex];
    }

    // returns: pair (offset, stride)
    std::pair<uint64_t, uint64_t> blockOffset(uint64_t rowBlockIndex,
                                              uint64_t colBlockIndex) const {
        // CHECK_GE(rowBlockIndex, colBlockIndex);
        uint64_t lump = spanToLump[colBlockIndex];
        uint64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
        uint64_t offsetInLump = spanOffsetInLump[colBlockIndex];
        uint64_t start = chainColPtr[lump];
        uint64_t end = chainColPtr[lump + 1];
        // bisect to find `rowBlockIndex` in chainRowSpan[start:end]
        uint64_t pos = bisect(chainRowSpan + start, end - start, rowBlockIndex);
        // CHECK_EQ(chainRowSpan[start + pos], rowBlockIndex);
        return std::make_pair(chainData[start + pos] + offsetInLump, lumpSize);
    }

    // returns: pair (offset, stride)
    std::pair<uint64_t, uint64_t> diagBlockOffset(uint64_t blockIndex) const {
        uint64_t lump = spanToLump[blockIndex];
        uint64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
        uint64_t offsetInLump = spanOffsetInLump[blockIndex];
        uint64_t start = chainColPtr[lump];
        return std::make_pair(chainData[start] + offsetInLump * (1 + lumpSize),
                              lumpSize);
    }

    const uint64_t* spanStart;
    const uint64_t* spanToLump;
    const uint64_t* lumpStart;
    const uint64_t* spanOffsetInLump;
    const uint64_t* chainColPtr;
    const uint64_t* chainRowSpan;
    const uint64_t* chainData;
};

struct PermutedCoalescedAccessor {
    PermutedCoalescedAccessor(const CoalescedAccessor& plainAcc,
                              const uint64_t* permutation)
        : plainAcc(plainAcc), permutation(permutation) {}

    uint64_t paramSize(uint64_t blockIndex) const {
        return plainAcc.paramSize(permutation[blockIndex]);
    }

    uint64_t paramStart(uint64_t blockIndex) const {
        return plainAcc.paramStart(permutation[blockIndex]);
    }

    std::tuple<uint64_t, uint64_t, bool> blockOffset(
        uint64_t rowBlockIndex, uint64_t colBlockIndex) const {
        uint64_t permRowBlockIndex = permutation[rowBlockIndex];
        uint64_t permColBlockIndex = permutation[colBlockIndex];
        auto [off, stride] = plainAcc.blockOffset(
            std::max(permRowBlockIndex, permColBlockIndex),
            std::min(permRowBlockIndex, permColBlockIndex));
        bool flipped = permRowBlockIndex < permColBlockIndex;
        return std::make_tuple(off, stride, flipped);
    }

    std::pair<uint64_t, uint64_t> diagBlockOffset(uint64_t blockIndex) {
        return plainAcc.diagBlockOffset(permutation[blockIndex]);
    }

    CoalescedAccessor plainAcc;
    const uint64_t* permutation;
};