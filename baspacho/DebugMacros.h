#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "baspacho/Utils.h"

// #define BASPACHO_CHECKS

#define BASPACHO_CHECK_WHAT(a, msg)                                            \
    if (!(a)) {                                                                \
        using namespace std::chrono;                                           \
        auto now = system_clock::now();                                        \
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;  \
        const std::time_t t_c = system_clock::to_time_t(now);                  \
        std::cerr << "[" << std::put_time(std::localtime(&t_c), "%T") << "."   \
                  << std::setfill('0') << std::setw(3) << ms.count()           \
                  << " " __FILE__ ":" << __LINE__ << "] Check failed: " << msg \
                  << std::endl;                                                \
        exit(1);                                                               \
    }

#ifdef BASPACHO_CHECKS
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
#define BASPACHO_CHECK(a) UNUSED(a)
#define BASPACHO_CHECK_OP(a, b, op) UNUSED(a, b)
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
