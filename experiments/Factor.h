#pragma once

#include "../baspacho/CoalescedBlockMatrix.h"

void factor(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data);

void factorLump(const CoalescedBlockMatrixSkel& skel, std::vector<double>& data,
                uint64_t aggreg);

void eliminateBoard(const CoalescedBlockMatrixSkel& skel,
                    std::vector<double>& data, uint64_t aggreg,
                    uint64_t rowItem);
