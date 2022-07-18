
#include <chrono>
#include <cmath>
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

// csv loading utility
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

// t ~= a + b*n + c*n^2 + d*n^3
void optimizePotrfModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 2);
  double n3coeff_sum = 0.0;
  for (auto& nt : samples) {
    double n = nt[0], t = nt[1];
    n3coeff_sum += log(t / (n * n * n));
  }
  double n3coeff = exp(n3coeff_sum / samples.size());

  using Vec4 = Eigen::Vector<double, 4>;
  using Mat14 = Eigen::Matrix<double, 1, 4>;
  Variable<Vec4> coeffs{0.0, 0.0, 0.0, n3coeff};

  Optimizer opt;
  for (auto& nt : samples) {
    double n = nt[0], t = nt[1];
    double s = 1.0 / t;  // scaling factor
    opt.addFactor(
        [=](const Vec4& c, Mat14* dc) -> Vec1 {
          if (dc) {
            (*dc)(0, 0) = s;
            (*dc)(0, 1) = n * s;
            (*dc)(0, 2) = n * n * s;
            (*dc)(0, 3) = n * n * n * s;
          }
          return Vec1(s * (c[0] + n * (c[1] + n * (c[2] + n * c[3])) - t));
        },
        coeffs);
  }

  opt.optimize();
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

  optimizePotrfModel(csvData);

  /*vector<Variable<Vec1>> pointVars = {{-2}, {-1}, {0}, {0.5}, {1.5}, {2.5}};

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

  opt.optimize();*/

  return 0;
}