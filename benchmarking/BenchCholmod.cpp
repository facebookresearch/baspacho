
#include "BenchCholmod.h"

#include <cholmod.h>

#include <chrono>
#include <iostream>
#include <random>

#include "../baspacho/DebugMacros.h"
#include "../baspacho/Utils.h"

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

std::pair<double, double> benchmarkCholmodSolve(
    const vector<uint64_t>& paramSize, const SparseStructure& ss,
    bool verbose) {
    BASPACHO_CHECK_EQ(paramSize.size(), ss.ptrs.size() - 1);
    vector<int64_t> rowPtr, colInd;
    vector<double> val;
    vector<uint64_t> spanStart = paramSize;
    spanStart.push_back(0);
    uint64_t totSize = cumSumVec(spanStart);
    rowPtr.push_back(0);

    mt19937 gen(37);
    uniform_real_distribution<double> unif(-1.0, 1.0);
    double diagBoost = totSize * 2;  // make positive definite

    if (verbose) {
        std::cout << "to csr... (order=" << totSize << ")" << std::endl;
    }
    for (uint64_t rb = 0; rb < paramSize.size(); rb++) {
        for (uint64_t ri = spanStart[rb]; ri < spanStart[rb + 1]; ri++) {
            // ri = row index
            for (uint64_t q = ss.ptrs[rb]; q < ss.ptrs[rb + 1]; q++) {
                uint64_t cb = ss.inds[q];
                for (uint64_t ci = spanStart[cb]; ci < spanStart[cb + 1];
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
    if (verbose) {
        std::cout << "to csr done." << std::endl;
    }

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

    if (verbose) {
        std::cout << "Analyzing..." << std::endl;
    }
    auto startAnalysis = hrc::now();
    cholmod_factor* cholmodFactor_ = cholmod_l_analyze(&A, &cc_);
    double analysisTime = tdelta(hrc::now() - startAnalysis).count();
    if (verbose) {
        std::cout << "Analysis time: " << analysisTime << "s" << std::endl;
    }
    BASPACHO_CHECK_EQ(cc_.status, CHOLMOD_OK);

    if (verbose) {
        std::cout << "Factoring..." << std::endl;
    }
    auto startFactor = hrc::now();
    const long oldPrintLevel = cc_.print;
    cc_.print = 1;
    cc_.quick_return_if_not_posdef = 1;
    long cholmod_status = cholmod_l_factorize(&A, cholmodFactor_, &cc_);
    cc_.print = oldPrintLevel;
    double factorTime = tdelta(hrc::now() - startFactor).count();
    if (verbose) {
        std::cout << "Factor time: " << factorTime << "s" << std::endl;
    }

    if (verbose) {
        std::cout << "Cholmod, A size: " << A.ncol << ", nz: " << A.nzmax
                  << " (" << A.nzmax / ((double)A.ncol * A.nrow) << " fill)"
                  << std::endl;
        if (!cholmodFactor_->is_super) {
            std::cout << "Cholmod, simplicial factor nz: "
                      << cholmodFactor_->nzmax << " ("
                      << cholmodFactor_->nzmax /
                             ((double)cholmodFactor_->n * cholmodFactor_->n)
                      << " fill)" << std::endl;
        } else {
            std::cout << "Cholmod\nsupernodes: " << cholmodFactor_->nsuper
                      << "\nssize: " << cholmodFactor_->ssize
                      << "\nxsize: " << cholmodFactor_->xsize
                      << "\nmaxcsize: " << cholmodFactor_->maxcsize
                      << "\nmaxesize: " << cholmodFactor_->maxesize << " ("
                      << cholmodFactor_->xsize /
                             ((double)cholmodFactor_->n * cholmodFactor_->n)
                      << " fill)" << std::endl;
            std::cout << "Stats:\ngemm calls: " << cc_.cholmod_cpu_gemm_calls
                      << ", time: " << cc_.cholmod_cpu_gemm_time
                      << "\nsyrk calls: " << cc_.cholmod_cpu_syrk_calls
                      << ", time: " << cc_.cholmod_cpu_syrk_time
                      << "\npotrf calls: " << cc_.cholmod_cpu_potrf_calls
                      << ", time: " << cc_.cholmod_cpu_potrf_time
                      << "\ntrsm calls: " << cc_.cholmod_cpu_trsm_calls
                      << ", time: " << cc_.cholmod_cpu_trsm_time
                      << "\nassemb1: " << cc_.cholmod_assemble_time
                      << ", assemb2: " << cc_.cholmod_assemble_time2
                      << std::endl;
        }
    }

    switch (cc_.status) {
        case CHOLMOD_NOT_INSTALLED:
            std::cerr << "CHOLMOD failure: Method not installed." << std::endl;
            exit(1);
        case CHOLMOD_OUT_OF_MEMORY:
            std::cerr << "CHOLMOD failure: Out of memory." << std::endl;
            exit(1);
        case CHOLMOD_TOO_LARGE:
            std::cerr << "CHOLMOD failure: Integer overflow occured."
                      << std::endl;
            exit(1);
        case CHOLMOD_INVALID:
            std::cerr << "CHOLMOD failure: Invalid input." << std::endl;
            exit(1);
        case CHOLMOD_NOT_POSDEF:
            std::cerr << "CHOLMOD warning: Matrix not positive definite."
                      << std::endl;
            exit(1);
        case CHOLMOD_DSMALL:
            std::cerr << "CHOLMOD warning: D for LDL' or diag(L) or LL' has "
                         "tiny absolute value."
                      << std::endl;
            exit(1);
        case CHOLMOD_OK:
            if (cholmod_status != 0) {
                if (verbose) {
                    std::cout << "Success!" << std::endl;
                }
                break;
            }
            std::cerr << "CHOLMOD failure: cholmod_factorize returned "
                         "false but cholmod_common::status "
                         "is CHOLMOD_OK."
                      << std::endl;
            exit(1);
        default:
            std::cerr << "Unknown cholmod return code: " << cc_.status
                      << std::endl;
            exit(1);
    }

    if (cholmodFactor_ != nullptr) {
        cholmod_l_free_factor(&cholmodFactor_, &cc_);
    }
    cholmod_l_finish(&cc_);

    return std::make_pair(analysisTime, factorTime);
}