#pragma once

#include "baspacho/Utils.h"

namespace BaSpaCho {

// returns all pairs (x, y) with 0 <= x <= y < n, while p varies in 0 <= p <
// n*(n+1)/2
__BASPACHO_HOST_DEVICE__ inline std::pair<int64_t, int64_t> toOrderedPair(
    int64_t n, int64_t p) {
    // the trick below converts p that varies in the range 0,1,...,n*(n+1)/2
    // to a pair x<=y, where y varies in the range 0,1,...,(n-1) and,
    // for each y, x varies in the range 0,1,...,y
    // furthermore, x increases sequentially in all pairs that are generated,
    // and this will optimize the memory accesses
    int64_t odd = n & 1;
    int64_t m = n + 1 - odd;
    int64_t x = p % m;            // here: x = 0,1,...,n-1
    int64_t y = n - 1 - (p / m);  // here: y = n-1,n-2,...,floor(n/2)
    if (x > y) {  // flip the triangle formed by points with x>y, and move it
                  // close to the origin
        x = x - y - 1;
        y = n - 1 - odd - y;
    }
    return std::make_pair(x, y);
}

template <typename T>
__BASPACHO_HOST_DEVICE__ inline static void cholesky(T* A, int n) {
    T* b_ii = A;

    for (int i = 0; i < n; i++) {
        T d = sqrt(*b_ii);
        *b_ii = d;

        // block(j, i)
        T* b_ji = b_ii + n;
        for (int j = i + 1; j < n; j++) {
            T c = *b_ji / d;
            *b_ji = c;

            T* b_ki = b_ii + n;
            T* b_jk = b_ji + 1;
            for (int k = i + 1; k <= j; k++) {
                *b_jk -= c * (*b_ki);
                b_ki += n;
                b_jk += 1;
            }

            b_ji += n;
        }

        b_ii += n + 1;
    }
}

template <typename T>
__BASPACHO_HOST_DEVICE__ inline static void solveUpperT(const T* A, int n,
                                                        T* v) {
    const T* b_ii = A;
    for (int i = 0; i < n; i++) {
        T x = v[i];

        for (int j = 0; j < i; j++) {
            x -= b_ii[j] * v[j];
        }

        v[i] = x / b_ii[i];
        b_ii += n;
    }
}

template <typename T>
__BASPACHO_HOST_DEVICE__ inline static void solveUpper(const T* A, int n,
                                                       T* v) {
    const T* b_ii = A + n * n - 1;
    for (int i = n - 1; i >= 0; i--) {
        T x = v[i];

        const T* b_ij = b_ii;
        for (int j = i + 1; j < n; j++) {
            b_ij += n;
            x -= (*b_ij) * v[j];
        }

        v[i] = x / (*b_ii);
        b_ii -= n + 1;
    }
}

}  // end namespace BaSpaCho