#pragma once

#include "BlockMatrix.h"

void factor(const BlockMatrixSkel& skel, std::vector<double>& data);

void factorAggreg(const BlockMatrixSkel& skel, std::vector<double>& data,
                  uint64_t aggreg);

void eliminateAggregItem(const BlockMatrixSkel& skel, std::vector<double>& data,
                         uint64_t aggreg, uint64_t rowItem);
