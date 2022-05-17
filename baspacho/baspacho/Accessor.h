
#include <Eigen/Geometry>
#include <tuple>

#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

// allows to retrieve a block in a coalesced-block matrix
struct CoalescedAccessor {
    // we don't define a constructor to be able to use it as argument of a Cuda kernel
    void init(const int64_t* spanStart_, const int64_t* spanToLump_,
              const int64_t* lumpStart_, const int64_t* spanOffsetInLump_,
              const int64_t* chainColPtr_, const int64_t* chainRowSpan_,
              const int64_t* chainData_) {
        spanStart = spanStart_;
        spanToLump = spanToLump_;
        lumpStart = lumpStart_;
        spanOffsetInLump = spanOffsetInLump_;
        chainColPtr = chainColPtr_;
        chainRowSpan = chainRowSpan_;
        chainData = chainData_;
    }

    __BASPACHO_HOST_DEVICE__
    int64_t paramSize(int64_t blockIndex) const {
        return spanStart[blockIndex + 1] - spanStart[blockIndex];
    }

    __BASPACHO_HOST_DEVICE__
    int64_t paramStart(int64_t blockIndex) const {
        return spanStart[blockIndex];
    }

    // returns: pair (offset, stride)
    __BASPACHO_HOST_DEVICE__
    std::pair<int64_t, int64_t> blockOffset(int64_t rowBlockIndex,
                                            int64_t colBlockIndex) const {
        BASPACHO_CHECK_GE(rowBlockIndex, colBlockIndex);
        int64_t lump = spanToLump[colBlockIndex];
        int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
        int64_t offsetInLump = spanOffsetInLump[colBlockIndex];
        int64_t start = chainColPtr[lump];
        int64_t end = chainColPtr[lump + 1];
        // bisect to find `rowBlockIndex` in chainRowSpan[start:end]
        int64_t pos = bisect(chainRowSpan + start, end - start, rowBlockIndex);
        BASPACHO_CHECK_EQ(chainRowSpan[start + pos], rowBlockIndex);
        return std::make_pair(chainData[start + pos] + offsetInLump, lumpSize);
    }

    // returns: pair (offset, stride)
    __BASPACHO_HOST_DEVICE__
    std::pair<int64_t, int64_t> diagBlockOffset(int64_t blockIndex) const {
        int64_t lump = spanToLump[blockIndex];
        int64_t lumpSize = lumpStart[lump + 1] - lumpStart[lump];
        int64_t offsetInLump = spanOffsetInLump[blockIndex];
        int64_t start = chainColPtr[lump];
        return std::make_pair(chainData[start] + offsetInLump * (1 + lumpSize),
                              lumpSize);
    }

    template <int rowSize = Eigen::Dynamic, int64_t colSize = Eigen::Dynamic,
              typename T>
    __BASPACHO_HOST_DEVICE__ auto block(T* data, int64_t rowBlockIndex,
                                        int64_t colBlockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = blockOffset(rowBlockIndex, colBlockIndex);
        if (rowSize != Dynamic) {
            BASPACHO_CHECK_EQ(rowSize, paramSize(rowBlockIndex));
        }
        if (colSize != Dynamic) {
            BASPACHO_CHECK_EQ(colSize, paramSize(colBlockIndex));
        }
        return Map<Matrix<T, rowSize, colSize, RowMajor>, 0, OuterStride<>>(
            data + offset,
            rowSize != Dynamic ? rowSize : paramSize(rowBlockIndex),
            colSize != Dynamic ? colSize : paramSize(colBlockIndex),
            OuterStride<>(stride));
    }

    template <int size = Eigen::Dynamic, typename T>
    __BASPACHO_HOST_DEVICE__ auto diagBlock(T* data, int64_t blockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = diagBlockOffset(blockIndex);
        if (size != Dynamic) {
            BASPACHO_CHECK_EQ(size, paramSize(blockIndex));
        }
        int pSize = size != Dynamic ? size : paramSize(blockIndex);
        return Map<Matrix<T, size, size, RowMajor>, 0, OuterStride<>>(
            data + offset, pSize, pSize, OuterStride<>(stride));
    }

    const int64_t* spanStart;
    const int64_t* spanToLump;
    const int64_t* lumpStart;
    const int64_t* spanOffsetInLump;
    const int64_t* chainColPtr;
    const int64_t* chainRowSpan;
    const int64_t* chainData;
};

// allows to retrieve a block in a coalesced-block matrix through a permutation
struct PermutedCoalescedAccessor {
    // we don't define a constructor to be able to use it as argument of a Cuda kernel
    void init(const CoalescedAccessor& plainAcc_, const int64_t* permutation_) {
        plainAcc = plainAcc_;
        permutation = permutation_;
    }

    // we don't define a constructor to be able to use it as argument of a Cuda kernel
    void init(const int64_t* spanStart_, const int64_t* spanToLump_,
              const int64_t* lumpStart_, const int64_t* spanOffsetInLump_,
              const int64_t* chainColPtr_, const int64_t* chainRowSpan_,
              const int64_t* chainData_, const int64_t* permutation_) {
        plainAcc.spanStart = spanStart_;
        plainAcc.spanToLump = spanToLump_;
        plainAcc.lumpStart = lumpStart_;
        plainAcc.spanOffsetInLump = spanOffsetInLump_;
        plainAcc.chainColPtr = chainColPtr_;
        plainAcc.chainRowSpan = chainRowSpan_;
        plainAcc.chainData = chainData_;
        permutation = permutation_;
    }

    __BASPACHO_HOST_DEVICE__
    int64_t paramSize(int64_t blockIndex) const {
        return plainAcc.paramSize(permutation[blockIndex]);
    }

    __BASPACHO_HOST_DEVICE__
    int64_t paramStart(int64_t blockIndex) const {
        return plainAcc.paramStart(permutation[blockIndex]);
    }

    __BASPACHO_HOST_DEVICE__
    std::tuple<int64_t, int64_t, bool> blockOffset(
        int64_t rowBlockIndex, int64_t colBlockIndex) const {
        int64_t permRowBlockIndex = permutation[rowBlockIndex];
        int64_t permColBlockIndex = permutation[colBlockIndex];
        auto [off, stride] = plainAcc.blockOffset(
            std::max(permRowBlockIndex, permColBlockIndex),
            std::min(permRowBlockIndex, permColBlockIndex));
        bool flipped = permRowBlockIndex < permColBlockIndex;
        return std::make_tuple(off, stride, flipped);
    }

    __BASPACHO_HOST_DEVICE__
    std::pair<int64_t, int64_t> diagBlockOffset(int64_t blockIndex) const {
        return plainAcc.diagBlockOffset(permutation[blockIndex]);
    }

    template <int rowSize = Eigen::Dynamic, int64_t colSize = Eigen::Dynamic,
              typename T>
    __BASPACHO_HOST_DEVICE__ auto block(T* data, int64_t rowBlockIndex,
                                        int64_t colBlockIndex) const {
        using namespace Eigen;
        if (rowSize != Dynamic) {
            BASPACHO_CHECK_EQ(rowSize, paramSize(rowBlockIndex));
        }
        if (colSize != Dynamic) {
            BASPACHO_CHECK_EQ(colSize, paramSize(colBlockIndex));
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
    __BASPACHO_HOST_DEVICE__ auto diagBlock(T* data, int64_t blockIndex) const {
        using namespace Eigen;
        auto [offset, stride] = diagBlockOffset(blockIndex);
        if (size != Dynamic) {
            BASPACHO_CHECK_EQ(size, paramSize(blockIndex));
        }
        int pSize = size != Dynamic ? size : paramSize(blockIndex);
        return Map<Matrix<T, size, size, RowMajor>, 0, OuterStride<>>(
            data + offset, pSize, pSize, OuterStride<>(stride));
    }

    CoalescedAccessor plainAcc;
    const int64_t* permutation;
};

}  // end namespace BaSpaCho