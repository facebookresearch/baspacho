#pragma once

#ifdef __CUDACC__
#define __BASPACHO_HOST_DEVICE__ __host__ __device__
#else
#define __BASPACHO_HOST_DEVICE__
#endif

namespace BaSpaCho {

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

}  // end namespace BaSpaCho