#pragma once

#include <iostream>

#include "baspacho/baspacho/Utils.h"

#define BASPACHO_CHECKS

#define BASPACHO_CHECK_WHAT(a, msg)                                      \
    if (!(a)) {                                                          \
        std::cerr << "[" << ::BaSpaCho::timeStamp() << " " __FILE__ ":"  \
                  << __LINE__ << "] Check failed: " << msg << std::endl; \
        exit(1);                                                         \
    }

#if defined(BASPACHO_CHECKS) && !defined(__CUDACC__)
#define BASPACHO_CHECK(a) BASPACHO_CHECK_WHAT(a, #a)
#define BASPACHO_CHECK_OP(a, b, op)                                          \
    {                                                                        \
        auto aEval = a;                                                      \
        auto bEval = b;                                                      \
        BASPACHO_CHECK_WHAT(aEval op bEval, #a " " #op " " #b " ("           \
                                                << aEval << " vs. " << bEval \
                                                << ")")                      \
    }
#else
#define BASPACHO_CHECK(a) ::BaSpaCho::UNUSED(a)
#define BASPACHO_CHECK_OP(a, b, op) ::BaSpaCho::UNUSED(a, b)
#endif

#define BASPACHO_CHECK_EQ(a, b) BASPACHO_CHECK_OP(a, b, ==)
#define BASPACHO_CHECK_LE(a, b) BASPACHO_CHECK_OP(a, b, <=)
#define BASPACHO_CHECK_LT(a, b) BASPACHO_CHECK_OP(a, b, <)
#define BASPACHO_CHECK_GE(a, b) BASPACHO_CHECK_OP(a, b, >=)
#define BASPACHO_CHECK_GT(a, b) BASPACHO_CHECK_OP(a, b, >)

#define BASPACHO_CHECK_NOTNULL(a)                                          \
    {                                                                      \
        auto aEval = a;                                                    \
        BASPACHO_CHECK_WHAT(aEval != nullptr, "'" #a "' Must be non NULL") \
    }
