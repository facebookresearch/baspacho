#pragma once

#include <type_traits>

// #define CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND
#ifdef CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND
// workaround for double on <6 architectures
[[maybe_unused]] __device__ static inline double atomicAdd(double* address,
                                                           double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(
            address_as_ull, assumed,
            ::__double_as_longlong(val + __longlong_as_double(assumed)));
        // integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// A -= B * C.T
template <typename A, typename B, typename C>
__device__ void locked_sub_product(A& aMat, const B& bMat, const C& cMatT) {
    for (int i = 0; i < bMat.rows(); i++) {
        for (int j = 0; j < cMatT.rows(); j++) {
            double* addr = &aMat(i, j);
            double val = -bMat.row(i).dot(cMatT.row(j));
            atomicAdd(addr, val);
        }
    }
}