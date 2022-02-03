
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include "baspacho/MathUtils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing;
using namespace std;

template <typename T>
using MatRMaj =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

TEST(MathUtils, Cholesky) {
    int n = 10;
    vector<double> data = randomData(n * n, -1.0, 1.0, 37);

    Eigen::Map<MatRMaj<double>>(data.data(), n, n).diagonal().array() +=
        n * 1.3;

    MatRMaj<double> verifyMat = Eigen::Map<MatRMaj<double>>(data.data(), n, n);
    cout << "orig:\n" << verifyMat << endl;

    { Eigen::LLT<Eigen::Ref<MatRMaj<double>>> llt(verifyMat); }
    cout << "verif:\n" << verifyMat << endl;

    cholesky(data.data(), n);
    MatRMaj<double> computedMat =
        Eigen::Map<MatRMaj<double>>(data.data(), n, n);
    cout << "comptd:\n" << computedMat << endl;

    ASSERT_NEAR(Eigen::MatrixXd(
                    (verifyMat - computedMat).triangularView<Eigen::Lower>())
                    .norm(),
                0, 1e-7);
}

TEST(MathUtils, Solve) {
    int n = 10, k = 1;
    vector<double> data = randomData(n * n, -1.0, 1.0, 37);
    Eigen::Map<MatRMaj<double>>(data.data(), n, n).diagonal().array() +=
        n * 0.3;
    Eigen::Map<MatRMaj<double>> verifyMat(data.data(), n, n);

    cout << "smat:\n" << verifyMat << endl;

    vector<double> vecData = randomData(n * k, -1.0, 1.0, 39);

    MatRMaj<double> verifyVec =
        Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);

    cout << "orig:\n" << verifyVec << endl;

    verifyMat.template triangularView<Eigen::Lower>()
        .transpose()
        .template solveInPlace<Eigen::OnTheRight>(verifyVec);

    cout << "verif:\n" << verifyVec << endl;

    solveUpperT(data.data(), n, vecData.data());
    MatRMaj<double> computeVec =
        Eigen::Map<MatRMaj<double>>(vecData.data(), k, n);
    cout << "comptd:\n" << computeVec << endl;

    ASSERT_NEAR((verifyVec - computeVec).norm(), 0, 1e-7);
}
