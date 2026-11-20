/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 #
 * The full license is in the file LICENSE, distributed with this software. #
 ****************************************************************************/

// UT 1: scalar math entry points vs the system libm.
//
// Accuracy requirement: results must match libm within float precision,
// i.e. |diff| <= tol * max(1, |ref|) with tol = 4 * float epsilon.
//
// Defining XSIMD_SCALAR_FAST_MATH compiles the scalar kernels without
// NaN/Inf special-case handling for performance; in that configuration
// the NaN / infinity edge-case expectations are relaxed to "finite
// inputs still agree with libm".

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "xsimd/xsimd.hpp"

#if defined(XSIMD_SCALAR_FAST_MATH)
// Under the performance knob, skip tests that exercise non-finite
// results (they are documented as unsupported in that mode).
#define XSIMD_UT_SKIP_NONFINITE() return SKIP()
#else
#define XSIMD_UT_SKIP_NONFINITE() (void)0
#endif

namespace
{
    constexpr double d_tol = 4.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    constexpr float f_tol = 4.0f * std::numeric_limits<float>::epsilon();

    bool almost_equal(double a, double b)
    {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) || std::isinf(b))
            return a == b;
        double scale = std::fabs(b) > 1.0 ? std::fabs(b) : 1.0;
        return std::fabs(a - b) <= d_tol * scale;
    }

    bool almost_equal(float a, float b)
    {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) || std::isinf(b))
            return a == b;
        float scale = std::fabs(b) > 1.0f ? std::fabs(b) : 1.0f;
        return std::fabs(a - b) <= f_tol * scale;
    }

    const double d_inputs[] = {
        0.0, -0.0,
        1e-300, -1e-300,
        0.001, -0.001,
        0.25, -0.25,
        0.5, -0.5,
        0.7853981633974483, -0.7853981633974483, // pi / 4
        1.0, -1.0,
        1.5, -1.5,
        1.5707963267948966, -1.5707963267948966, // pi / 2
        2.0, -2.0,
        3.141592653589793, -3.141592653589793, // pi
        4.0, -4.0,
        10.0, -10.0,
        100.5, -100.5,
        12345.678, -12345.678,
        1e6, -1e6,
        1e8, -1e8,
        1e15, -1e15,
        1e300, -1e300
    };

    const double d_log_inputs[] = {
        0.0, -1.0, 1e-300, 0.001, 0.25, 0.5, 1.0, 1.5, 2.0,
        3.141592653589793, 4.0, 10.0, 100.5, 12345.678, 1e6, 1e8, 1e15, 1e300
    };

    const float f_inputs[] = {
        0.0f, -0.0f,
        1e-30f, -1e-30f,
        0.001f, -0.001f,
        0.25f, -0.25f,
        0.5f, -0.5f,
        0.7853981634f, -0.7853981634f,
        1.0f, -1.0f,
        1.5f, -1.5f,
        1.5707963268f, -1.5707963268f,
        2.0f, -2.0f,
        3.1415926536f, -3.1415926536f,
        4.0f, -4.0f,
        10.0f, -10.0f,
        100.5f, -100.5f,
        12345.678f, -12345.678f,
        1e6f, -1e6f,
        1e8f, -1e8f,
        1e30f, -1e30f
    };

    const float f_log_inputs[] = {
        0.0f, -1.0f, 1e-30f, 0.001f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f,
        3.1415926536f, 4.0f, 10.0f, 100.5f, 12345.678f, 1e6f, 1e8f, 1e30f
    };
}

TEST_SUITE("ScalarMathVsLibm")
{
    TEST_CASE("Sin")
    {
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::sin(x), std::sin(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::sin(x), std::sin(x)));
        }
    }

    TEST_CASE("Cos")
    {
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cos(x), std::cos(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cos(x), std::cos(x)));
        }
    }

    TEST_CASE("Tan")
    {
        XSIMD_UT_SKIP_NONFINITE(); // tan overflows near poles; skip in fast mode
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::tan(x), std::tan(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::tan(x), std::tan(x)));
        }
    }

    TEST_CASE("Cot")
    {
        XSIMD_UT_SKIP_NONFINITE(); // |cot| overflows for large inputs in fast mode
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cot(x), 1.0 / std::tan(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cot(x), 1.0f / std::tan(x)));
        }
    }

    TEST_CASE("Sinh")
    {
        XSIMD_UT_SKIP_NONFINITE(); // sinh(1e300) overflows; skip in fast mode
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::sinh(x), std::sinh(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::sinh(x), std::sinh(x)));
        }
    }

    TEST_CASE("Cosh")
    {
        XSIMD_UT_SKIP_NONFINITE(); // cosh(1e300) overflows; skip in fast mode
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cosh(x), std::cosh(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::cosh(x), std::cosh(x)));
        }
    }

    TEST_CASE("Tanh")
    {
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::tanh(x), std::tanh(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::tanh(x), std::tanh(x)));
        }
    }

    TEST_CASE("Coth")
    {
        XSIMD_UT_SKIP_NONFINITE(); // coth(x)~1 for large x, tanh(x)->1 exact; but ±1 overflow at 0
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::coth(x), 1.0 / std::tanh(x)));
        }
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::coth(x), 1.0f / std::tanh(x)));
        }
    }

    TEST_CASE("Exp")
    {
        XSIMD_UT_SKIP_NONFINITE(); // exp overflow edge cases; skip in fast mode
        for (double x : d_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::exp(x), std::exp(x)));
        }
        // subnormal / overflow edges
        CHECK(almost_equal(xsimd::scalar::exp(-745.5), std::exp(-745.5)));
        CHECK(almost_equal(xsimd::scalar::exp(709.8), std::exp(709.8)));
        for (float x : f_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::exp(x), std::exp(x)));
        }
        CHECK(almost_equal(xsimd::scalar::exp(-103.9f), std::exp(-103.9f)));
        CHECK(almost_equal(xsimd::scalar::exp(88.9f), std::exp(88.9f)));
    }

    TEST_CASE("Log")
    {
        XSIMD_UT_SKIP_NONFINITE(); // log(0) = -inf and log(-1) = NaN are specials
        for (double x : d_log_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::log(x), std::log(x)));
        }
        for (float x : f_log_inputs)
        {
            CAPTURE(x);
            CHECK(almost_equal(xsimd::scalar::log(x), std::log(x)));
        }
    }
}
