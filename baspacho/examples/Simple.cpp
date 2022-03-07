
#include <chrono>
#include <iomanip>

#include "Optimizer.h"
#include "baspacho/baspacho/Solver.h"
#include "baspacho/baspacho/Utils.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing;
using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;
using Vec1 = Eigen::Vector<double, 1>;
using Mat11 = Eigen::Matrix<double, 1, 1>;

int main(int argc, char* argv[]) {
    Optimizer opt;

    vector<Variable<Vec1>> pointVars(6);
    pointVars[0].value[0] = -2;
    pointVars[1].value[0] = -1;
    pointVars[2].value[0] = -0;
    pointVars[3].value[0] = 0.5;
    pointVars[4].value[0] = 1.5;
    pointVars[5].value[0] = 2.5;

    for (size_t i = 0; i < 5; i++) {
        opt.addFactor(
            [](const Vec1& x, const Vec1& y, Mat11* dx, Mat11* dy) -> Vec1 {
                if (dx) {
                    (*dx)(0, 0) = -1;
                }
                if (dy) {
                    (*dy)(0, 0) = 1;
                }
                return Vec1(y[0] - x[0] - 1);
            },
            pointVars[i], pointVars[i + 1]);
    }

    opt.optimize();

    return 0;
}