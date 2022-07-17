
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "Optimizer.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing;
using namespace std;
using Vec1 = Eigen::Vector<double, 1>;
using Mat11 = Eigen::Matrix<double, 1, 1>;

vector<vector<double>> loadCsv(const string& path) {
  ifstream iStr(path);
  if (!iStr) {
    throw runtime_error("could not open file'" + path + "'");
  }

  string line;
  vector<vector<double>> retv;
  while (getline(iStr, line)) {
    vector<double> entries;
    stringstream lStr(line);
    string entry;
    while (getline(lStr, entry, '\t')) {
      if (entry.empty()) {
        throw runtime_error("empty entry while expecting numeric entry");
      }
      size_t nChars = 0;
      double nEntry = stod(entry, &nChars);
      if (nChars != entry.size()) {
        throw runtime_error("unable to convert all characters while processing entry '" + entry +
                            "'");
      }
      entries.push_back(nEntry);
    }
    if (!retv.empty() && retv[0].size() != entries.size()) {
      throw runtime_error("inconsistent row length in CSV file");
    }
    retv.push_back(std::move(entries));
  }
  return retv;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: opt_comp_model file.csv";
  }
  auto csvData = loadCsv(argv[1]);
  if (csvData.empty()) {
    throw runtime_error("Empty csv file!");
  }
  cout << "CSV data have " << csvData[0].size() << " entries per line" << endl;

  vector<Variable<Vec1>> pointVars = {{-2}, {-1}, {0}, {0.5}, {1.5}, {2.5}};

  Optimizer opt;
  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    static constexpr double springLen = 1.0;
    opt.addFactor(
        [=](const Vec1& x, const Vec1& y, Mat11* dx, Mat11* dy) -> Vec1 {
          if (dx) {
            (*dx)(0, 0) = -1;
          }
          if (dy) {
            (*dy)(0, 0) = 1;
          }
          return Vec1(y[0] - x[0] - springLen);
        },
        pointVars[i], pointVars[i + 1]);
  }

  opt.optimize();

  return 0;
}