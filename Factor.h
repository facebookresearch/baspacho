#pragma once

#include "BlockMatrix.h"

void factor(const BlockMatrixSkel& skel, std::vector<double>& data);

void factorLump(const BlockMatrixSkel& skel, std::vector<double>& data,
                uint64_t aggreg);

void eliminateBoard(const BlockMatrixSkel& skel, std::vector<double>& data,
                    uint64_t aggreg, uint64_t rowItem);
