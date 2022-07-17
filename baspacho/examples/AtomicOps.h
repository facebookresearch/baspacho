#pragma once

// magic IEEE double number locking mechanism
// there is a magic value (a NaN with "content" 0xb10cd, "blocked")
// which his functionally equivalent to a NaN but cannot be ever
// obtained as result of a computation, and can therefore be used as
// a "magic" value to block matrix of double/float numbers
template <typename T>
struct IEEEMagicLock;

template <>
struct IEEEMagicLock<double> {
  // this is a magic value (a NaN with "content" 0xb10cd, "blocked")
  // which his functionally equivalent to a NaN but cannot be ever
  // obtained as result of a computation, and can therefore be used as
  // a "magic" value to block matrix of double numbers
  // For future reference a magic value for floats cuold be 0x7f80b1cdu
  static constexpr uint64_t magicValue = 0x7ff00000000b10cdul;

  static inline __attribute__((always_inline)) double lock(double* address) {
    uint64_t retv;
    do {
      retv = __atomic_exchange_n((uint64_t*)address, magicValue, __ATOMIC_ACQUIRE);
    } while (retv == magicValue);
    return *(double*)(&retv);
  }

  static inline __attribute__((always_inline)) void unlock(double* address, double val) {
    __atomic_store_n((uint64_t*)address, *(uint64_t*)(&val), __ATOMIC_RELEASE);
  }
};

template <>
struct IEEEMagicLock<float> {
  static constexpr uint32_t magicValue = 0x7f80b1cdu;

  static inline __attribute__((always_inline)) float lock(float* address) {
    uint32_t retv;
    do {
      retv = __atomic_exchange_n((uint32_t*)address, magicValue, __ATOMIC_ACQUIRE);
    } while (retv == magicValue);
    return *(float*)(&retv);
  }

  static inline __attribute__((always_inline)) void unlock(float* address, float val) {
    __atomic_store_n((uint32_t*)address, *(uint32_t*)(&val), __ATOMIC_RELEASE);
  }
};

// u += v for matrices/vectors, using "magic locking"
struct LockedSharedOps {
  template <typename U, typename V>
  static inline __attribute__((always_inline)) void matrixAdd(U& u, const V& v) {
    using T = std::remove_reference_t<decltype(u[0])>;
    T u00 = IEEEMagicLock<T>::lock(&u(0, 0)) + v(0, 0);  // lock u(0,0)
    for (int j = 1; j < u.cols(); j++) {                 // add all other elements
      u(0, j) += v(0, j);
    }
    for (int i = 1; i < u.rows(); i++) {
      for (int j = 0; j < u.cols(); j++) {
        u(i, j) += v(i, j);
      }
    }
    IEEEMagicLock<T>::unlock(&u(0, 0), u00);  // unlock u(0,0) writing new value
  }

  template <typename U, typename V>
  static inline __attribute__((always_inline)) void vectorAdd(U& u, const V& v) {
    using T = std::remove_reference_t<decltype(u[0])>;
    T u0 = IEEEMagicLock<T>::lock(&u[0]) + v[0];  // lock u[0]
    for (int j = 1; j < u.size(); j++) {          // add all other elements
      u[j] += v[j];
    }
    IEEEMagicLock<T>::unlock(&u[0], u0);  // unlock u[0] writing new value
  }
};

// dummy ops (for single-threaded operation)
struct PlainOps {
  template <typename U, typename V>
  static inline __attribute__((always_inline)) void matrixAdd(U& u, const V& v) {
    u += v;
  }

  template <typename U, typename V>
  static inline __attribute__((always_inline)) void vectorAdd(U& u, const V& v) {
    u += v;
  }
};