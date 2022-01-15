
#include "BenchCholmod.h"

#include <cholmod.h>
#include <glog/logging.h>

#include <chrono>
#include <random>

#include "Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

void benchmarkCholmodSolve(const vector<uint64_t>& paramSize,
                           const SparseStructure& ss) {
    CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
    vector<int64_t> rowPtr, colInd;
    vector<double> val;
    vector<uint64_t> paramStart = paramSize;
    paramStart.push_back(0);
    uint64_t totSize = cumSum(paramStart);
    rowPtr.push_back(0);

    mt19937 gen(37);
    uniform_real_distribution<double> unif(-1.0, 1.0);
    double diagBoost = totSize * 2;  // make positive definite

    LOG(INFO) << "to csr... (order=" << totSize << ")";
    for (uint64_t rb = 0; rb < paramSize.size(); rb++) {
        for (uint64_t ri = paramStart[rb]; ri < paramStart[rb + 1]; ri++) {
            // ri = row index
            for (uint64_t q = ss.ptrs[rb]; q < ss.ptrs[rb + 1]; q++) {
                uint64_t cb = ss.inds[q];
                for (uint64_t ci = paramStart[cb]; ci < paramStart[cb + 1];
                     ci++) {
                    // ci = col index
                    if (ci > ri) {
                        continue;
                    }
                    colInd.push_back(ci);
                    val.push_back(unif(gen) + (ci == ri ? diagBoost : 0.0));
                }
            }
            rowPtr.push_back(colInd.size());
        }
    }
    LOG(INFO) << "to csr done.";

    cholmod_common cc_;
    cholmod_l_start(&cc_);
    cc_.nmethods = 1;
    cc_.method[0].ordering = CHOLMOD_AMD;
    cc_.supernodal = CHOLMOD_AUTO;
    // cc_.supernodal = CHOLMOD_SIMPLICIAL;
    // cc_.supernodal = CHOLMOD_SUPERNODAL;

    cholmod_sparse A;
    A.nzmax = val.size();
    A.nrow = rowPtr.size() - 1;
    A.ncol = rowPtr.size() - 1;
    A.p = rowPtr.data();
    A.i = colInd.data();
    A.x = val.data();
    A.sorted = 1;
    A.packed = 1;

    // 0: not symmetric, applies factorization to A*At.
    // 1: symmetric, stores upper triangular part,
    // -1: lower triangular
    A.stype = 1;  // as compressed-column, it's upper-tri
    A.itype = CHOLMOD_LONG;
    A.xtype = CHOLMOD_REAL;
    A.dtype = CHOLMOD_DOUBLE;

    LOG(INFO) << "Analyzing...";
    auto startAnalysis = hrc::now();
    cholmod_factor* cholmodFactor_ = cholmod_l_analyze(&A, &cc_);
    LOG(INFO) << "Analysis time: " << tdelta(hrc::now() - startAnalysis).count()
              << "s";
    CHECK_EQ(cc_.status, CHOLMOD_OK)
        << "cholmod_analyze failed. error code: " << cc_.status;

    LOG(INFO) << "Factoring...";
    auto startFactor = hrc::now();
    const long oldPrintLevel = cc_.print;
    cc_.print = 0;
    cc_.quick_return_if_not_posdef = 1;
    long cholmod_status = cholmod_l_factorize(&A, cholmodFactor_, &cc_);
    cc_.print = oldPrintLevel;
    LOG(INFO) << "Factor time: " << tdelta(hrc::now() - startFactor).count()
              << "s";

    LOG(INFO) << "Cholmod, A size: " << A.ncol << ", nz: " << A.nzmax << " ("
              << A.nzmax / ((double)A.ncol * A.nrow) << " fill)";
    if (!cholmodFactor_->is_super) {
        LOG(INFO) << "Cholmod, simplicial factor nz: " << cholmodFactor_->nzmax
                  << " ("
                  << cholmodFactor_->nzmax /
                         ((double)cholmodFactor_->n * cholmodFactor_->n)
                  << " fill)";
    } else {
        LOG(INFO) << "Cholmod, supernodes: " << cholmodFactor_->nsuper
                  << ", ssize: " << cholmodFactor_->ssize
                  << ", xsize: " << cholmodFactor_->xsize
                  << ", maxcsize: " << cholmodFactor_->maxcsize
                  << ", maxesize: " << cholmodFactor_->maxesize << " ("
                  << cholmodFactor_->xsize /
                         ((double)cholmodFactor_->n * cholmodFactor_->n)
                  << " fill)";
    }

    switch (cc_.status) {
        case CHOLMOD_NOT_INSTALLED:
            LOG(FATAL) << "CHOLMOD failure: Method not installed.";
        case CHOLMOD_OUT_OF_MEMORY:
            LOG(FATAL) << "CHOLMOD failure: Out of memory.";
        case CHOLMOD_TOO_LARGE:
            LOG(FATAL) << "CHOLMOD failure: Integer overflow occured.";
        case CHOLMOD_INVALID:
            LOG(FATAL) << "CHOLMOD failure: Invalid input.";
        case CHOLMOD_NOT_POSDEF:
            LOG(FATAL) << "CHOLMOD warning: Matrix not positive definite.";
        case CHOLMOD_DSMALL:
            LOG(FATAL) << "CHOLMOD warning: D for LDL' or diag(L) or LL' has "
                          "tiny absolute value.";
        case CHOLMOD_OK:
            if (cholmod_status != 0) {
                LOG(INFO) << "Success!";
                break;
            }
            LOG(FATAL) << "CHOLMOD failure: cholmod_factorize returned "
                          "false but cholmod_common::status "
                          "is CHOLMOD_OK.";
        default:
            LOG(FATAL) << "Unknown cholmod return code: " << cc_.status;
    }

    if (cholmodFactor_ != nullptr) {
        cholmod_l_free_factor(&cholmodFactor_, &cc_);
    }
    cholmod_l_finish(&cc_);
}