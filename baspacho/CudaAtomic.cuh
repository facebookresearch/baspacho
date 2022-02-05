#pragma once

#include <type_traits>

/*
  Select the strategy to write atomically a block. Possibilities are:
  1. Magic block lock: first entry is atomically replaced with a magic
    NaN value (if wasn't already, in such case retry). This is a NaN
    with "contect" that cannot be confused with a NaN coming from an
    arithmetic operation, and acts as a lock while other entries are
    written. Finally first entry is updated as well.
    Note that this on Cuda is a bit tricky for reasons related to thread
    scheduling, as explained in the relevant section below.
  2. Atomically add all entries using atomicAdd. This requires Cuda
    Compute architecture >=6 for double type (>=2 for float).
  TODO: neither strategy looks ideal, keep investigating if there is any
  better option available.
*/
#define USE_MAGIC_BLOCK_LOCK

#ifdef USE_MAGIC_BLOCK_LOCK
/*
  `value` is a magic value (a NaN with "content" 0xb10cd, "blocked")
   which is functionally equivalent to a NaN but cannot be ever
   obtained as result of a computation, and can therefore be used as
   a "magic" value to block matrix of float/double numbers
*/
template <typename T>
struct Magic;

template <>
struct Magic<double> {
    using type = unsigned long long int;
    static constexpr type value = 0x7ff00000000b10cdULL;
};

template <>
struct Magic<float> {
    using type = unsigned int;
    static constexpr type value = 0x7f80b1cdU;
};

// A -= B * C.T
template <typename A, typename B, typename C>
__device__ void locked_sub_product(A& aMat, const B& bMat, const C& cMatT) {
    using T = typename std::remove_reference<decltype(aMat(0, 0))>::type;
    using I = typename Magic<T>::type;
    static constexpr I magicValue = Magic<T>::value;
    /*
      Tricky! cannot exchange loop order because of scheduling in same warp, or
      one thread may be scheduled in an infinite loop while the locking thread
      is not scheduled and unable to unlock, as explained in:
        https://forums.developer.nvidia.com/t/try-to-use-lock-and-unlock-in-cuda/50761
      and expecially in:
        https://stackoverflow.com/questions/31194291/cuda-mutex-why-deadlock
    */
    T* lockPtr = &aMat(0, 0);
    do {
        I oldVal = atomicExch((I*)lockPtr, magicValue);
        if (oldVal == magicValue) {
            continue;
        }
        T lockVal = (T&)oldVal - bMat.row(0).dot(cMatT.row(0));
        for (int j = 1; j < cMatT.rows(); j++) {
            aMat(0, j) -= bMat.row(0).dot(cMatT.row(j));
        }
        for (int i = 1; i < bMat.rows(); i++) {
            for (int j = 0; j < cMatT.rows(); j++) {
                aMat(i, j) -= bMat.row(i).dot(cMatT.row(j));
            }
        }
        /*
          Value write is atomic, see eg:
          https://forums.developer.nvidia.com/t/which-write-operations-are-atomic-in-cuda/54019
        */
        *lockPtr = lockVal;
    } while (0);
}

#else  // USE_MAGIC_BLOCK_LOCK

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
#endif  // USE_MAGIC_BLOCK_LOCK