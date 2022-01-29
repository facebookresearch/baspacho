#include "CoalescedBlockMatrix.h"

#include <glog/logging.h>

#include <iostream>

#include "Utils.h"

using namespace std;

CoalescedBlockMatrixSkel::CoalescedBlockMatrixSkel(
    const vector<uint64_t>& spanStart, const vector<uint64_t>& lumpToSpan,
    const vector<uint64_t>& colPtr, const vector<uint64_t>& rowInd)
    : spanStart(spanStart), lumpToSpan(lumpToSpan) {
    CHECK_GE(spanStart.size(), lumpToSpan.size());
    CHECK_GE(lumpToSpan.size(), 1);
    CHECK_EQ(spanStart.size() - 1, lumpToSpan[lumpToSpan.size() - 1]);
    CHECK_EQ(colPtr.size(), lumpToSpan.size());
    CHECK(isStrictlyIncreasing(spanStart, 0, spanStart.size()));
    CHECK(isStrictlyIncreasing(lumpToSpan, 0, lumpToSpan.size()));

    uint64_t totSize = spanStart[spanStart.size() - 1];
    uint64_t numSpans = spanStart.size() - 1;
    uint64_t numLumps = lumpToSpan.size() - 1;

    spanToLump.resize(numSpans);
    lumpStart.resize(numLumps + 1);
    for (size_t l = 0; l < numLumps; l++) {
        uint64_t sBegin = lumpToSpan[l];
        uint64_t sEnd = lumpToSpan[l + 1];
        lumpStart[l] = spanStart[sBegin];
        for (size_t s = sBegin; s < sEnd; s++) {
            spanToLump[s] = l;
        }
    }
    lumpStart[numLumps] = totSize;
    spanOffsetInLump.resize(numSpans);
    for (uint64_t s = 0; s < numSpans; s++) {
        spanOffsetInLump[s] = spanStart[s] - lumpStart[spanToLump[s]];
    }

    chainColPtr.resize(numLumps + 1);
    chainRowSpan.clear();
    chainData.clear();

    boardColPtr.resize(numLumps + 1);
    boardRowLump.clear();
    boardChainColOrd.clear();
    uint64_t dataPtr = 0;
    uint64_t gatheredDataPtr = 0;
    for (size_t l = 0; l < numLumps; l++) {
        uint64_t colStart = colPtr[l];
        uint64_t colEnd = colPtr[l + 1];
        CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
        uint64_t lSpanBegin = lumpToSpan[l];
        uint64_t lSpanEnd = lumpToSpan[l + 1];
        uint64_t lSpanSize = lSpanEnd - lSpanBegin;
        uint64_t lDataSize = lumpStart[l + 1] - lumpStart[l];

        // check the initial section is the set of params from `a`, and
        // therefore the full diagonal block is contained in the matrix
        CHECK_GE(colEnd - colStart, lSpanSize)
            << "Column must contain full diagonal block";
        CHECK_EQ(rowInd[colStart], lSpanBegin)
            << "Column data must start at diagonal block";
        CHECK_EQ(rowInd[colStart + lSpanSize - 1], lSpanEnd - 1)
            << "Column must contain full diagonal block";

        chainColPtr[l] = chainRowSpan.size();
        boardColPtr[l] = boardRowLump.size();
        uint64_t currentRowAggreg = kInvalid;
        uint64_t numRowsSkipped = 0;
        for (size_t i = colStart; i < colEnd; i++) {
            uint64_t p = rowInd[i];
            chainRowSpan.push_back(p);
            chainData.push_back(dataPtr);
            dataPtr += lDataSize * (spanStart[p + 1] - spanStart[p]);
            numRowsSkipped += spanStart[p + 1] - spanStart[p];
            chainRowsTillEnd.push_back(numRowsSkipped);

            uint64_t rowAggreg = spanToLump[p];
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
    for (size_t l = 0; l < numLumps; l++) {
        for (uint64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
            uint64_t rowLump = boardRowLump[i];
            boardRowPtr[rowLump]++;
        }
    }
    uint64_t numBoards = cumSumVec(boardRowPtr);
    boardColLump.resize(numBoards);
    boardColOrd.resize(numBoards);
    for (size_t l = 0; l < numLumps; l++) {
        for (uint64_t i = boardColPtr[l]; i < boardColPtr[l + 1] - 1; i++) {
            uint64_t rowLump = boardRowLump[i];
            boardColLump[boardRowPtr[rowLump]] = l;
            boardColOrd[boardRowPtr[rowLump]] = i - boardColPtr[l];
            boardRowPtr[rowLump]++;
        }
    }
    rewindVec(boardRowPtr);
}

Eigen::MatrixXd CoalescedBlockMatrixSkel::densify(
    const std::vector<double>& data) {
    uint64_t totData = chainData[chainData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = spanStart[spanStart.size() - 1];
    Eigen::MatrixXd retv(totSize, totSize);
    retv.setZero();

    for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
        uint64_t lBegin = lumpStart[a];
        uint64_t lSize = lumpStart[a + 1] - lBegin;
        uint64_t colStart = chainColPtr[a];
        uint64_t colEnd = chainColPtr[a + 1];
        for (uint64_t i = colStart; i < colEnd; i++) {
            uint64_t p = chainRowSpan[i];
            uint64_t pStart = spanStart[p];
            uint64_t pSize = spanStart[p + 1] - pStart;
            uint64_t dataPtr = chainData[i];

            retv.block(pStart, lBegin, pSize, lSize) =
                Eigen::Map<const MatRMaj<double>>(data.data() + dataPtr, pSize,
                                                  lSize);
        }
    }

    return retv;
}

void CoalescedBlockMatrixSkel::damp(std::vector<double>& data, double alpha,
                                    double beta) {
    uint64_t totData = chainData[chainData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = spanStart[spanStart.size() - 1];

    for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
        uint64_t aStart = lumpStart[a];
        uint64_t aSize = lumpStart[a + 1] - aStart;
        uint64_t colStart = chainColPtr[a];
        uint64_t dataPtr = chainData[colStart];

        Eigen::Map<MatRMaj<double>> block(data.data() + dataPtr, aSize, aSize);
        block.diagonal() *= (1 + alpha);
        block.diagonal().array() += beta;
    }
}
