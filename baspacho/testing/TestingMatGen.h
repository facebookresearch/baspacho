#pragma once

#include <random>
#include <set>
#include <vector>

namespace BaSpaCho::testing {

//
struct SparseMatGenerator {
  SparseMatGenerator(int64_t size, int64_t seed = 37);

  void connectRanges(int64_t begin1, int64_t end1, int64_t begin2, int64_t end2,
                     double fill,
                     int64_t maxOffset = std::numeric_limits<int64_t>::max());

  void addSparseConnections(double fill);

  void addSchurSet(int64_t size, double fill);

  static SparseMatGenerator genFlat(int64_t size, double fill,
                                    int64_t seed = 37);

  // topology is roughly a line, entries in band are set with a probability
  static SparseMatGenerator genLine(int64_t size, double fill, int64_t bandSize,
                                    int64_t seed = 37);

  // topology is a set of meridians (connecting north and south poles)
  static SparseMatGenerator genMeridians(int64_t num, int64_t lineLen,
                                         double fill, int64_t bandSize,
                                         int64_t hairLen, int64_t nPoleHairs,
                                         int64_t sPoleHairs, int64_t seed = 37);

  // generates parameters in a grid, with connection to neighbors
  static SparseMatGenerator genGrid(int64_t width, int64_t height, double fill,
                                    int64_t connMaxDist, int64_t seed = 37);

  std::mt19937 gen;
  std::vector<std::set<int64_t>> columns;
};

}  // end namespace BaSpaCho::testing