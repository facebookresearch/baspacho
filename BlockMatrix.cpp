#include "BlockMatrix.h"

#include <glog/logging.h>

#include <iostream>

#include "Utils.h"

using namespace std;

BlockMatrixSkel::BlockMatrixSkel(const vector<uint64_t>& spanStart,
                                 const vector<uint64_t>& lumpToSpan,
                                 const vector<uint64_t>& colPtr,
                                 const vector<uint64_t>& rowInd)
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
    for (size_t a = 0; a < numLumps; a++) {
        uint64_t aStart = lumpToSpan[a];
        uint64_t aEnd = lumpToSpan[a + 1];
        lumpStart[a] = spanStart[aStart];
        for (size_t i = aStart; i < aEnd; i++) {
            spanToLump[i] = a;
        }
    }
    lumpStart[numLumps] = totSize;

    chainColPtr.resize(numLumps + 1);
    chainRowSpan.clear();
    chainData.clear();

    boardColPtr.resize(numLumps + 1);
    boardRowLump.clear();
    boardChainColOrd.clear();
    uint64_t dataPtr = 0;
    uint64_t gatheredDataPtr = 0;
    for (size_t a = 0; a < numLumps; a++) {
        uint64_t colStart = colPtr[a];
        uint64_t colEnd = colPtr[a + 1];
        CHECK(isStrictlyIncreasing(rowInd, colStart, colEnd));
        uint64_t aParamStart = lumpToSpan[a];
        uint64_t aParamEnd = lumpToSpan[a + 1];
        uint64_t aParamSize = aParamEnd - aParamStart;
        uint64_t aDataSize = lumpStart[a + 1] - lumpStart[a];

        // check the initial section is the set of params from `a`, and
        // therefore the full diagonal block is contained in the matrix
        CHECK_GE(colEnd - colStart, aParamSize)
            << "Column must contain full diagonal block";
        CHECK_EQ(rowInd[colStart], aParamStart)
            << "Column data must start at diagonal block";
        CHECK_EQ(rowInd[colStart + aParamSize - 1], aParamEnd - 1)
            << "Column must contain full diagonal block";

        chainColPtr[a] = chainRowSpan.size();
        boardColPtr[a] = boardRowLump.size();
        uint64_t currentRowAggreg = kInvalid;
        uint64_t numRowsSkipped = 0;
        for (size_t i = colStart; i < colEnd; i++) {
            uint64_t p = rowInd[i];
            chainRowSpan.push_back(p);
            chainData.push_back(dataPtr);
            dataPtr += aDataSize * (spanStart[p + 1] - spanStart[p]);
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

    /*
    std::vector<uint64_t> boardRowPtr;
    std::vector<uint64_t> boardColLump;
    std::vector<uint64_t> boardChainColOrd;
    */

    boardRowPtr.assign(numLumps + 1, 0);
    for (size_t a = 0; a < numLumps; a++) {
        for (uint64_t i = boardColPtr[a]; i < boardColPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = boardRowLump[i];
            boardRowPtr[rowAggreg]++;
        }
    }
    uint64_t numBoards = cumSum(boardRowPtr);
    boardColLump.resize(numBoards);
    boardColOrd.resize(numBoards);
    for (size_t a = 0; a < numLumps; a++) {
        for (uint64_t i = boardColPtr[a]; i < boardColPtr[a + 1] - 1; i++) {
            uint64_t rowAggreg = boardRowLump[i];
            boardColLump[boardRowPtr[rowAggreg]] = a;
            boardColOrd[boardRowPtr[rowAggreg]] = i - boardColPtr[a];
            boardRowPtr[rowAggreg]++;
        }
    }
    rewind(boardRowPtr);
}

Eigen::MatrixXd BlockMatrixSkel::densify(const std::vector<double>& data) {
    uint64_t totData = chainData[chainData.size() - 1];
    CHECK_EQ(totData, data.size());

    uint64_t totSize = spanStart[spanStart.size() - 1];
    Eigen::MatrixXd retv(totSize, totSize);
    retv.setZero();

    for (size_t a = 0; a < chainColPtr.size() - 1; a++) {
        uint64_t aStart = lumpStart[a];
        uint64_t aSize = lumpStart[a + 1] - aStart;
        uint64_t colStart = chainColPtr[a];
        uint64_t colEnd = chainColPtr[a + 1];
        for (uint64_t i = colStart; i < colEnd; i++) {
            uint64_t p = chainRowSpan[i];
            uint64_t pStart = spanStart[p];
            uint64_t pSize = spanStart[p + 1] - pStart;
            uint64_t dataPtr = chainData[i];

            retv.block(pStart, aStart, pSize, aSize) =
                Eigen::Map<const MatRMaj<double>>(data.data() + dataPtr, pSize,
                                                  aSize);
        }
    }

    return retv;
}

void BlockMatrixSkel::damp(std::vector<double>& data, double alpha,
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
