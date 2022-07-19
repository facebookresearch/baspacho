
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
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec4& c, Mat14* dc) -> Vec1 {
          if (dc) {
            *dc << s, n * s, n * n * s, n * n * n * s;
          }
          return Vec1(s * (c[0] + n * (c[1] + n * (c[2] + n * c[3])) - t));
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "potrf_model = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << " };";
  cout << ss.str() << endl;
}

// t ~= a + b*n + c*n^2 + (d + e*n + f*n^2)*k
void optimizeTrsmModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 3);
  double n2k_coeff_sum = 0.0;
  for (auto& nkt : samples) {
    double n = nkt[0], k = nkt[1], t = nkt[2];
    n2k_coeff_sum += log(t / (n * n * k));
  }
  double n2k_coeff = exp(n2k_coeff_sum / samples.size());

  using Vec6 = Eigen::Vector<double, 6>;
  using Mat16 = Eigen::Matrix<double, 1, 6>;
  Variable<Vec6> coeffs{0.0, 0.0, 0.0, 0.0, 0.0, n2k_coeff};

  Optimizer opt;
  for (auto& nkt : samples) {
    double n = nkt[0], k = nkt[1], t = nkt[2];
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec6& c, Mat16* dc) -> Vec1 {
          if (dc) {
            *dc << s, n * s, n * n * s, k * s, k * n * s, k * n * n * s;
          }
          return Vec1(s * (c[0] + n * (c[1] + n * c[2]) + k * (c[3] + n * (c[4] + n * c[5])) - t));
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "trsm_model = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ", " << c[4]
     << ", " << c[5] << " };";
  cout << ss.str() << endl;
}

// plain model is:
//   t ~= a + b*m + c*n + d*k + e*m*n + f*m*k + g*n*k + h*n*m*k
// symmetrized in m,n it becomes (putting u=m+n, v=mn the basis of sym functions):
//   t ~= a + b*u + c*v + d*k + e*u*k + f*v*k
void optimizeSygeModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 4);
  double mnk_coeff_sum = 0.0;
  for (auto& mnkt : samples) {
    double m = mnkt[0], n = mnkt[1], k = mnkt[2], t = mnkt[3];
    mnk_coeff_sum += log(t / (m * n * k));
  }
  double mnk_coeff = exp(mnk_coeff_sum / samples.size());

  using Vec6 = Eigen::Vector<double, 6>;
  using Mat16 = Eigen::Matrix<double, 1, 6>;
  Variable<Vec6> coeffs{0.0, 0.0, 0.0, 0.0, 0.0, mnk_coeff};

  Optimizer opt;
  for (auto& mnkt : samples) {
    double m = mnkt[0], n = mnkt[1], k = mnkt[2], t = mnkt[3];
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec6& c, Mat16* dc) -> Vec1 {
          if (dc) {
            *dc << s, (m + n) * s, (m * n) * s, k * s, k * (m + n) * s, k * (m * n) * s;
          }
          return Vec1(s * (c[0] + (m + n) * c[1] + (m * n) * c[2] +  //
                           k * (c[3] + (m + n) * c[4] + (m * n) * c[5]) - t));
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "syge_model = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ", " << c[4]
     << ", " << c[5] << " };";
  cout << ss.str() << endl;
}

// t ~= a + b*br + c*bc + d*br*bc
void optimizeAsmblModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 3);
  double brbc_coeff_sum = 0.0;
  for (auto& brbct : samples) {
    double br = brbct[0], bc = brbct[1], t = brbct[2];
    brbc_coeff_sum += log(t / (br * bc));
  }
  double brbc_coeff = exp(brbc_coeff_sum / samples.size());

  using Vec4 = Eigen::Vector<double, 4>;
  using Mat14 = Eigen::Matrix<double, 1, 4>;
  Variable<Vec4> coeffs{0.0, 0.0, 0.0, brbc_coeff};

  Optimizer opt;
  for (auto& brbct : samples) {
    double br = brbct[0], bc = brbct[1], t = brbct[2];
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec4& c, Mat14* dc) -> Vec1 {
          if (dc) {
            *dc << s, br * s, bc * s, br * bc * s;
          }
          return Vec1(s * (c[0] + br * c[1] + bc * c[2] + br * bc * c[3] - t));
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "asmbl_model = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << " };";
  cout << ss.str() << endl;
}

int main(int argc, char* argv[]) {
  std::string model, input;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m")) {
      if (i == argc - 1) {
        cout << "missing arg to -m argument" << endl;
        return 1;
      }
      model = argv[++i];
    } else if (!strcmp(argv[i], "-i")) {
      if (i == argc - 1) {
        cout << "missing arg to -i argument" << endl;
        return 1;
      }
      input = argv[++i];
    } else if (!strcmp(argv[i], "-h")) {
      cout << "Usage: -m potrf|trsm|syge|asmbl -i input_file" << endl;
      return 0;
    } else {
      cout << "Unknown arg '" << argv[i] << "' (-h for help)" << endl;
      return 1;
    }
  }
  if (model.empty() || input.empty()) {
    cout << "missing model/input arguments (-h for help)" << endl;
    return 1;
  }

  auto csvData = loadCsv(input);
  if (csvData.empty()) {
    throw runtime_error("Empty csv file!");
  }
  cout << "loaded CSV data have " << csvData[0].size() << " entries per line" << endl;

  if (model == "potrf") {
    optimizePotrfModel(csvData);
  } else if (model == "trsm") {
    optimizeTrsmModel(csvData);
  } else if (model == "syge") {
    optimizeSygeModel(csvData);
  } else if (model == "asmbl") {
    optimizeAsmblModel(csvData);
  } else {
    cout << "unknown model " << model << endl;
    return 1;
  }

  return 0;
}