/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 #
 * The full license is in the file LICENSE, distributed with this software. #
 ****************************************************************************/

// UT 2: scalar math entry points vs the batch kernels.
//
// Requirements verified here:
//  * results are BIT-IDENTICAL to the corresponding batch computation
//    (memcmp on the object representation, NaNs included);
//  * the test binary does not reference any libm symbol (checked by the
//    accompanying CMake target, which links without libm and greps the
//    dynamic symbol table).
//
// Defining XSIMD_SCALAR_FAST_MATH compiles the scalar kernels without
// NaN/Inf special-case handling for performance; the bit-exactness
// checks are SKIPPED entirely in that configuration.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#if defined(XSIMD_SCALAR_FAST_MATH)
TEST_SUITE("ScalarMathVsBatch")
{
    TEST_CASE("SkippedUnderFastMath")
    {
        SKIP();
    }
}
#else

#include <cstring>

#include "xsimd/xsimd.hpp"

// doctest's ConsoleReporter formats elapsed time with log10; provide the
// symbol from the libm-free xsimd scalar kernel so nothing is pulled
// from libm in this fully static binary.
extern "C" double log10(double x) noexcept
{
    return xsimd::scalar::log10(x);
}

namespace
{
    template <class T>
    bool same_bits(T a, T b)
    {
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }

    const double inputs[] = {
        0.0, -0.0,
        1e-300, -1e-300,
        0.001, -0.001,
        0.25, -0.25,
        0.5, -0.5,
        0.7853981633974483, -0.7853981633974483, // pi/4
        1.0, -1.0,
        1.5, -1.5,
        1.5707963267948966, -1.5707963267948966, // pi/2
        2.0, -2.0,
        3.141592653589793, -3.141592653589793, // pi
        4.0, -4.0,
        10.0, -10.0,
        62.83185307179586, -62.83185307179586, // 20*pi (reduction branch edge)
        100.5, -100.5,
        12345.678, -12345.678,
        1e6, -1e6,
        1e8, -1e8,
        1e15, -1e15,
        1e300, -1e300
    };

    const double log_inputs[] = {
        0.0, -1.0, 1e-300, 0.001, 0.25, 0.5, 1.0, 1.5, 2.0,
        3.141592653589793, 4.0, 10.0, 100.5, 12345.678, 1e6, 1e8, 1e15, 1e300
    };
}

#define SCALAR_BATCH_BIT_TEST(Func, InputArray)                                     \
    TEST_SUITE("ScalarMathVsBatchDouble")                                           \
    {                                                                               \
        TEST_CASE(#Func)                                                            \
        {                                                                           \
            for (double x : InputArray)                                             \
            {                                                                       \
                CAPTURE(x);                                                         \
                CHECK(same_bits(xsimd::scalar::Func(x),                             \
                                  xsimd::Func(xsimd::batch<double>(x)).get(0)));     \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    TEST_SUITE("ScalarMathVsBatchFloat")                                            \
    {                                                                               \
        TEST_CASE(#Func)                                                            \
        {                                                                           \
            for (double x : InputArray)                                             \
            {                                                                       \
                CAPTURE(x);                                                         \
                CHECK(same_bits(xsimd::scalar::Func(static_cast<float>(x)),          \
                                  xsimd::Func(xsimd::batch<float>(static_cast<float>(x))).get(0))); \
            }                                                                       \
        }                                                                           \
    }

SCALAR_BATCH_BIT_TEST(sin, inputs)
SCALAR_BATCH_BIT_TEST(cos, inputs)
SCALAR_BATCH_BIT_TEST(tan, inputs)
SCALAR_BATCH_BIT_TEST(sinh, inputs)
SCALAR_BATCH_BIT_TEST(cosh, inputs)
SCALAR_BATCH_BIT_TEST(tanh, inputs)
SCALAR_BATCH_BIT_TEST(exp, inputs)
SCALAR_BATCH_BIT_TEST(log, log_inputs)

// cot / coth have no batch counterpart; they are defined as 1/tan and
// 1/tanh respectively on both sides, so compare against that.
TEST_SUITE("ScalarMathVsBatchDouble")
{
    TEST_CASE("cot")
    {
        for (double x : inputs)
        {
            CAPTURE(x);
            CHECK(same_bits(xsimd::scalar::cot(x),
                            1.0 / xsimd::tan(xsimd::batch<double>(x)).get(0)));
        }
    }

    TEST_CASE("coth")
    {
        for (double x : inputs)
        {
            CAPTURE(x);
            CHECK(same_bits(xsimd::scalar::coth(x),
                            1.0 / xsimd::tanh(xsimd::batch<double>(x)).get(0)));
        }
    }
}

TEST_SUITE("ScalarMathVsBatchFloat")
{
    TEST_CASE("cot")
    {
        for (double x : inputs)
        {
            CAPTURE(x);
            CHECK(same_bits(xsimd::scalar::cot(static_cast<float>(x)),
                            1.0f / xsimd::tan(xsimd::batch<float>(static_cast<float>(x))).get(0)));
        }
    }

    TEST_CASE("coth")
    {
        for (double x : inputs)
        {
            CAPTURE(x);
            CHECK(same_bits(xsimd::scalar::coth(static_cast<float>(x)),
                            1.0f / xsimd::tanh(xsimd::batch<float>(static_cast<float>(x))).get(0)));
        }
    }
}

#endif // XSIMD_SCALAR_FAST_MATH
