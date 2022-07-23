
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "Optimizer.h"
#include "baspacho/baspacho/ComputationModel.h"
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

using Vec4 = Eigen::Vector<double, 4>;
using Mat14 = Eigen::Matrix<double, 1, 4>;
using Vec6 = Eigen::Vector<double, 6>;
using Mat16 = Eigen::Matrix<double, 1, 6>;

// t ~= a + b*n + c*n^2 + d*n^3
Vec4 optimizePotrfModel(const vector<vector<double>>& samples) {
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
          dc && (*dc = ComputationModel::dPotrfModel(n) * s, true);
          return Vec1{s * (ComputationModel::potrfModel(c, n) - t)};
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "potrfParams = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << " };";
  cout << ss.str() << endl;

  return coeffs.value;
}

// t ~= a + b*n + c*n^2 + (d + e*n + f*n^2)*k
Vec6 optimizeTrsmModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 3);
  double n2k_coeff_sum = 0.0;
  for (auto& nkt : samples) {
    double n = nkt[0], k = nkt[1], t = nkt[2];
    n2k_coeff_sum += log(t / (n * n * k));
  }
  double n2k_coeff = exp(n2k_coeff_sum / samples.size());

  Variable<Vec6> coeffs{0.0, 0.0, 0.0, 0.0, 0.0, n2k_coeff};

  Optimizer opt;
  for (auto& nkt : samples) {
    double n = nkt[0], k = nkt[1], t = nkt[2];
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec6& c, Mat16* dc) -> Vec1 {
          dc && (*dc = ComputationModel::dTrsmModel(n, k) * s, true);
          return Vec1{s * (ComputationModel::trsmModel(c, n, k) - t)};
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "trsmParams = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ", " << c[4]
     << ", " << c[5] << " };";
  cout << ss.str() << endl;

  return coeffs.value;
}

// plain model is:
//   t ~= a + b*m + c*n + d*k + e*m*n + f*m*k + g*n*k + h*n*m*k
// symmetrized in m,n it becomes (putting u=m+n, v=mn the basis of sym functions):
//   t ~= a + b*u + c*v + d*k + e*u*k + f*v*k
Vec6 optimizeSygeModel(const vector<vector<double>>& samples) {
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
          dc && (*dc = ComputationModel::dSygeModel(m, n, k) * s, true);
          return Vec1{s * (ComputationModel::sygeModel(c, m, n, k) - t)};
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "sygeParams = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ", " << c[4]
     << ", " << c[5] << " };";
  cout << ss.str() << endl;

  return coeffs.value;
}

// t ~= a + b*br + c*bc + d*br*bc
Vec4 optimizeAsmblModel(const vector<vector<double>>& samples) {
  BASPACHO_CHECK_EQ(samples[0].size(), 3);
  double brbc_coeff_sum = 0.0;
  for (auto& brbct : samples) {
    double br = brbct[0], bc = brbct[1], t = brbct[2];
    brbc_coeff_sum += log(t / (br * bc));
  }
  double brbc_coeff = exp(brbc_coeff_sum / samples.size());

  Variable<Vec4> coeffs{0.0, 0.0, 0.0, brbc_coeff};

  Optimizer opt;
  for (auto& brbct : samples) {
    double br = brbct[0], bc = brbct[1], t = brbct[2];
    double s = 1.0 / sqrt(t);  // scaling factor
    opt.addFactor(
        [=](const Vec4& c, Mat14* dc) -> Vec1 {
          dc && (*dc = ComputationModel::dAsmblModel(br, bc) * s, true);
          return Vec1{s * (ComputationModel::asmblModel(c, br, bc) - t)};
        },
        coeffs);
  }

  opt.optimize();

  stringstream ss;
  ss.precision(numeric_limits<double>::max_digits10 + 2);
  auto& c = coeffs.value;
  ss << "asmblParams = { " << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << " };";
  cout << ss.str() << endl;

  return coeffs.value;
}

int main(int argc, char* argv[]) {
  std::string potrf, trsm, syge, asmbl;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-p")) {
      if (i == argc - 1) {
        cout << "missing arg to -p argument (run with -h)" << endl;
        return 1;
      }
      potrf = argv[++i];
    } else if (!strcmp(argv[i], "-t")) {
      if (i == argc - 1) {
        cout << "missing arg to -t argument (run with -h)" << endl;
        return 1;
      }
      trsm = argv[++i];
    } else if (!strcmp(argv[i], "-g")) {
      if (i == argc - 1) {
        cout << "missing arg to -g argument (run with -h)" << endl;
        return 1;
      }
      syge = argv[++i];
    } else if (!strcmp(argv[i], "-a")) {
      if (i == argc - 1) {
        cout << "missing arg to -a argument (run with -h)" << endl;
        return 1;
      }
      asmbl = argv[++i];
    } else if (!strcmp(argv[i], "-h")) {
      cout << "Usage: -p potrf.csv -t trsm.csv -g syge.csv -a asmbl.csv" << endl;
      return 0;
    } else {
      cout << "Unknown arg '" << argv[i] << "' (-h for help)" << endl;
      return 1;
    }
  }
  if (potrf.empty() && trsm.empty() && syge.empty() && asmbl.empty()) {
    cout << "no csv file given! (-h for help)" << endl;
    return 1;
  }

  ComputationModel model;

  if (!potrf.empty()) {
    auto csvData = loadCsv(potrf);
    BASPACHO_CHECK(!csvData.empty());
    model.potrfParams = optimizePotrfModel(csvData);
  }

  if (!trsm.empty()) {
    auto csvData = loadCsv(trsm);
    BASPACHO_CHECK(!csvData.empty());
    model.trsmParams = optimizeTrsmModel(csvData);
  }

  if (!syge.empty()) {
    auto csvData = loadCsv(syge);
    BASPACHO_CHECK(!csvData.empty());
    model.sygeParams = optimizeSygeModel(csvData);
  }

  if (!asmbl.empty()) {
    auto csvData = loadCsv(asmbl);
    BASPACHO_CHECK(!csvData.empty());
    model.asmblParams = optimizeAsmblModel(csvData);
  }

  if (!potrf.empty() && !trsm.empty() && !syge.empty() && !asmbl.empty()) {
    stringstream ss;
    ss.precision(numeric_limits<double>::max_digits10 + 2);
    ss << "\n\nCopy & paste computation model code:\n"
       << "BaSpaCho::ComputationModel myModel {\n"
       << "  { " << model.potrfParams[0] << ", " << model.potrfParams[1] << ", "
       << model.potrfParams[2] << ", " << model.potrfParams[3] << "},\n"
       << "  { " << model.trsmParams[0] << ", " << model.trsmParams[1] << ", "
       << model.trsmParams[2] << ", " << model.trsmParams[3] << ", " << model.trsmParams[4] << ", "
       << model.trsmParams[5] << "},\n"
       << "  { " << model.sygeParams[0] << ", " << model.sygeParams[1] << ", "
       << model.sygeParams[2] << ", " << model.sygeParams[3] << ", " << model.sygeParams[4] << ", "
       << model.sygeParams[5] << "},\n"
       << "  { " << model.asmblParams[0] << ", " << model.asmblParams[1] << ", "
       << model.asmblParams[2] << ", " << model.asmblParams[3] << "}\n"
       << "};";
    cout << ss.str() << endl;
  }

  return 0;
}