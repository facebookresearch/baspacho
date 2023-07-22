/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include "baspacho/baspacho/Utils.h"

#ifndef NO_BASPACHO_CHECKS
#define BASPACHO_CHECKS
#endif  // NO_BASPACHO_CHECKS

#define BASPACHO_CHECK_WHAT1(a, msg)                 \
  if (!(a)) {                                        \
    ::BaSpaCho::throwError(__FILE__, __LINE__, msg); \
  }

#define BASPACHO_CHECK_WHAT2(a, what, v1, v2)                 \
  if (!(a)) {                                                 \
    ::BaSpaCho::throwError(__FILE__, __LINE__, what, v1, v2); \
  }

#if defined(BASPACHO_CHECKS) && !defined(__CUDACC__)
#define BASPACHO_CHECK(a) BASPACHO_CHECK_WHAT1(a, #a)
#define BASPACHO_CHECK_OP(a, b, op)                                       \
  {                                                                       \
    auto aEval = a;                                                       \
    auto bEval = b;                                                       \
    BASPACHO_CHECK_WHAT2(aEval op bEval, #a " " #op " " #b, aEval, bEval) \
  }
#else
#define BASPACHO_CHECK(a) ::BaSpaCho::BASPACHO_UNUSED(a)
#define BASPACHO_CHECK_OP(a, b, op) ::BaSpaCho::BASPACHO_UNUSED(a, b)
#endif

#define BASPACHO_CHECK_EQ(a, b) BASPACHO_CHECK_OP(a, b, ==)
#define BASPACHO_CHECK_LE(a, b) BASPACHO_CHECK_OP(a, b, <=)
#define BASPACHO_CHECK_LT(a, b) BASPACHO_CHECK_OP(a, b, <)
#define BASPACHO_CHECK_GE(a, b) BASPACHO_CHECK_OP(a, b, >=)
#define BASPACHO_CHECK_GT(a, b) BASPACHO_CHECK_OP(a, b, >)

#define BASPACHO_CHECK_NOTNULL(a)                                       \
  {                                                                     \
    auto aEval = a;                                                     \
    BASPACHO_CHECK_WHAT1(aEval != nullptr, "'" #a "' Must be non NULL") \
  }
