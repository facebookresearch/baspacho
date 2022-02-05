#include "baspacho/CoalescedBlockMatrix.h"

#include <iostream>

#include "baspacho/DebugMacros.h"
#include "baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;

CoalescedBlockMatrixSkel::CoalescedBlockMatrixSkel(
    const vector<int64_t>& spanStart, const vector<int64_t>& lumpToSpan,
    const vector<int64_t>& colPtr, const vector<int64_t>& rowInd)
    : spanStart(spanStart), lumpToSpan(lumpToSpan) {
    BASPACHO_CHECK_GE(spanStart.size(), lumpToSpan.size());
    BASPACHO_CHECK_GE((int64_t)lumpToSpan.size(), 1);
    BASPACHO_CHECK_EQ((int64_t)spanStart.size() - 1,
                      lumpToSpan[lumpToSpan.size() - 1]);
    BASPACHO_CHECK_EQ(colPtr.size(), lumpToSpan.size());
    BASPACHO_CHECK(isStrictlyIncreasing(spanStart, 0, spanStart.size()));
    BASPACHO_CHECK(isStrictlyIncreasing(lumpToSpan, 0, lumpToSpan.size()));

    int64_t totSize = spanStart[spanStart.size() - 1];
    int64_t numSpans = spanStart.size() - 1;
    int64_t numLumps = lumpToSpan.size() - 1;

    spanToLump.resize(numSpans);
    lumpStart.resize(numLumps + 1);
    for (int64_t l = 0; l < numLumps; l++) {
        int64_t sBegin = lumpToSpan[l];
        int64_t sEnd = lumpToSpan[l + 1];
        lumpStart[l] = spanStart[sBegin];
        for (int64_t s = sBegin; s < sEnd; s++) {
            spanToLump[s] = l;
        }
    }
    lumpStart[numLumps] = totSize;
    spanOffsetInLump.resize(numSpans);
    for (int64_t s = 0; s < numSpans; s++) {
        spanOffsetInLump[s] = spanStart[s] - lumpStart[spanToLump[s]];
    }

    chainColPtr.resize(numLumps + 1);
    chainRowSpan.clear();
    chainData.clear();

    boardColPtr.resize(numLumps + 1);
    boardRowLump.clear();
    boardChainColOrd.clear();
    int64_t dataPtr = 0;
    for (int64_t l = 0; l < numLumps; l++) {
        int64_t colStart = colPtr[l];
        int64_t colEnd = colPtr[l + 1];
        BASPACHO_CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
        int64_t lSpanBegin = lumpToSpan[l];
        int64_t lSpanEnd = lumpToSpan[l + 1];
        int64_t lSpanSize = lSpanEnd - lSpanBegin;
        int64_t lDataSize = lumpStart[l + 1] - lumpStart[l];

        // check the initial section is the set of params from `a`, and
        // therefore the full diagonal block is contained in the matrix
        // Column must contain full diagonal block:
        BASPACHO_CHECK_GE(colEnd - colStart, lSpanSize);
        // Column data must start at diagonal block:
        BASPACHO_CHECK_EQ(rowInd[colStart], lSpanBegin);
        // Column must contain full diagonal block:
        BASPACHO_CHECK_EQ(rowInd[colStart + lSpanSize - 1], lSpanEnd - 1);

        chainColPtr[l] = chainRowSpan.size();
        boardColPtr[l] = boardRowLump.size();
        int64_t currentRowAggreg = kInvalid;
        int64_t numRowsSkipped = 0;
        for (int64_t i = colStart; i < colEnd; i++) {
            int64_t p = rowInd[i];
            chainRowSpan.push_back(p);
            chainData.push_back(dataPtr);
            dataPtr += lDataSize * (spanStart[p + 1] - spanStart[p]);
            numRowsSkipped += spanStart[p + 1] - spanStart[p];
            chainRowsTillEnd.push_back(numRowsSkipped);

            int64_t rowAggreg = spanToLump[p];
            if (rowAggreg != currentRowAggreg) {
                currentRowAggreg = rowAggreg;
                boardRowLump.push_back(rowAggreg);
                boardChainColOrd.push_back(i - colStart);
            }
        }
        boardRowLump.push_back(kInvalid);
        boardChainColOrd.push_back(colEnd - colStart);
    }
    chainColPtr[numLumps] = chainRowSpan.size();
    boardColPtr[numLumps] = boardRowLump.size();
    chainData.push_back(dataPtr);

    boardRowPtr.assign(numLumps + 1, 0);
    for (int64_t l = 0; l < numLumps; l++) {
        for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
            int64_t rowLump = boardRowLump[i];
            boardRowPtr[rowLump]++;
        }
    }
    int64_t numBoards = cumSumVec(boardRowPtr);
    boardColLump.resize(numBoards);
    boardColOrd.resize(numBoards);
    for (int64_t l = 0; l < numLumps; l++) {
        for (int64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
            int64_t rowLump = boardRowLump[i];
            boardColLump[boardRowPtr[rowLump]] = l;
            boardColOrd[boardRowPtr[rowLump]] = i - boardColPtr[l];
            boardRowPtr[rowLump]++;
        }
    }
    rewindVec(boardRowPtr);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify(const std::vector<T>& data) const {
    int64_t totData = chainData[chainData.size() - 1];
    BASPACHO_CHECK_EQ(totData, (int64_t)data.size());

    int64_t totSize = spanStart[spanStart.size() - 1];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> retv(totSize, totSize);
    retv.setZero();

    for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
        int64_t lBegin = lumpStart[a];
        int64_t lSize = lumpStart[a + 1] - lBegin;
        int64_t colStart = chainColPtr[a];
        int64_t colEnd = chainColPtr[a + 1];
        for (int64_t i = colStart; i < colEnd; i++) {
            int64_t p = chainRowSpan[i];
            int64_t pStart = spanStart[p];
            int64_t pSize = spanStart[p + 1] - pStart;
            int64_t dataPtr = chainData[i];

            retv.block(pStart, lBegin, pSize, lSize) =
                Eigen::Map<const MatRMaj<T>>(data.data() + dataPtr, pSize,
                                             lSize);
        }
    }

    return retv;
}

template <typename T>
void CoalescedBlockMatrixSkel::damp(std::vector<T>& data, T alpha,
                                    T beta) const {
    int64_t totData = chainData[chainData.size() - 1];
    BASPACHO_CHECK_EQ(totData, (int64_t)data.size());

    for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
        int64_t aStart = lumpStart[a];
        int64_t aSize = lumpStart[a + 1] - aStart;
        int64_t colStart = chainColPtr[a];
        int64_t dataPtr = chainData[colStart];

        Eigen::Map<MatRMaj<T>> block(data.data() + dataPtr, aSize, aSize);
        block.diagonal() *= (1 + alpha);
        block.diagonal().array() += beta;
    }
}

template Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<double>(
    const std::vector<double>& data) const;
template Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
CoalescedBlockMatrixSkel::densify<float>(const std::vector<float>& data) const;
template void CoalescedBlockMatrixSkel::damp<double>(std::vector<double>& data,
                                                     double alpha,
                                                     double beta) const;
template void CoalescedBlockMatrixSkel::damp<float>(std::vector<float>& data,
                                                    float alpha,
                                                    float beta) const;

}  // end namespace BaSpaCho