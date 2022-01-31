
#include <glog/logging.h>

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

    template <int rowSize = Eigen::Dynamic, int64_t colSize = Eigen::Dynamic,
              typename T>
    auto block(T* data, uint64_t rowBlockIndex, uint64_t colBlockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = blockOffset(rowBlockIndex, colBlockIndex);
        if (rowSize != Dynamic) {
            CHECK_EQ(rowSize, paramSize(rowBlockIndex));
        }
        if (colSize != Dynamic) {
            CHECK_EQ(colSize, paramSize(colBlockIndex));
        }
        return Map<Matrix<T, rowSize, colSize, RowMajor>, 0, OuterStride<>>(
            data + offset,
            rowSize != Dynamic ? rowSize : paramSize(rowBlockIndex),
            colSize != Dynamic ? colSize : paramSize(colBlockIndex),
            OuterStride<>(stride));
    }

    template <int size = Eigen::Dynamic, typename T>
    auto diagBlock(T* data, uint64_t blockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = diagBlockOffset(blockIndex);
        if (size != Dynamic) {
            CHECK_EQ(size, paramSize(blockIndex));
        }
        int pSize = size != Dynamic ? size : paramSize(blockIndex);
        return Map<Matrix<T, size, size, RowMajor>, 0, OuterStride<>>(
            data + offset, pSize, pSize, OuterStride<>(stride));
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

    std::pair<uint64_t, uint64_t> diagBlockOffset(uint64_t blockIndex) const {
        return plainAcc.diagBlockOffset(permutation[blockIndex]);
    }

    template <int rowSize = Eigen::Dynamic, int64_t colSize = Eigen::Dynamic,
              typename T>
    auto block(T* data, uint64_t rowBlockIndex, uint64_t colBlockIndex) const {
        using namespace Eigen;
        if (rowSize != Dynamic) {
            CHECK_EQ(rowSize, paramSize(rowBlockIndex));
        }
        if (colSize != Dynamic) {
            CHECK_EQ(colSize, paramSize(colBlockIndex));
        }
        auto [offset, stride, flip] = blockOffset(rowBlockIndex, colBlockIndex);
        return Map<Matrix<T, rowSize, colSize, RowMajor>, 0,
                   Stride<Dynamic, Dynamic>>(
            data + offset,
            rowSize != Dynamic ? rowSize : paramSize(rowBlockIndex),
            colSize != Dynamic ? colSize : paramSize(colBlockIndex),
            Stride<Dynamic, Dynamic>(flip ? 1 : stride, flip ? stride : 1));
    }

    template <int size = Eigen::Dynamic, typename T>
    auto diagBlock(T* data, uint64_t blockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = diagBlockOffset(blockIndex);
        if (size != Dynamic) {
            CHECK_EQ(size, paramSize(blockIndex));
        }
        int pSize = size != Dynamic ? size : paramSize(blockIndex);
        return Map<Matrix<T, size, size, RowMajor>, 0, OuterStride<>>(
            data + offset, pSize, pSize, OuterStride<>(stride));
    }

    CoalescedAccessor plainAcc;
    const uint64_t* permutation;
};