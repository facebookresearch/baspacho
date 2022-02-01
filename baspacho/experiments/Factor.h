#pragma once

#include "baspacho/CoalescedBlockMatrix.h"

namespace BaSpaCho {

void factor(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data);

void factorLump(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data,
                int64_t aggreg);

void eliminateBoard(const CoalescedBlockMatrixSkel& skel,
                    std::vector<double>& data, int64_t aggreg, int64_t rowItem);

}  // end namespace BaSpaCho