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

#define BASPACHO_CHECK_WHAT(a, msg)                                                             \
  if (!(a)) {                                                                                   \
    throw std::runtime_error((std::stringstream()                                               \
                              << "[" << ::BaSpaCho::timeStamp() << " " __FILE__ ":" << __LINE__ \
                              << "] Check failed: " << msg)                                     \
                                 .str());                                                       \
  }

#if defined(BASPACHO_CHECKS) && !defined(__CUDACC__)
#define BASPACHO_CHECK(a) BASPACHO_CHECK_WHAT(a, #a)
#define BASPACHO_CHECK_OP(a, b, op)                                                 \
  {                                                                                 \
    auto aEval = a;                                                                 \
    auto bEval = b;                                                                 \
    BASPACHO_CHECK_WHAT(aEval op bEval,                                             \
                        #a " " #op " " #b " (" << aEval << " vs. " << bEval << ")") \
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

#define BASPACHO_CHECK_NOTNULL(a)                                      \
  {                                                                    \
    auto aEval = a;                                                    \
    BASPACHO_CHECK_WHAT(aEval != nullptr, "'" #a "' Must be non NULL") \
  }
