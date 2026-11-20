/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 #
 *                                                                          #
 * Distributed under the terms of the BSD 3-Clause License.                 #
 * The full license is in the file LICENSE, distributed with this software. #
 ****************************************************************************/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include "xsimd/xsimd.hpp"

TEST_SUITE("ScalarArithmetic")
{
    TEST_CASE("AddSubMulDiv")
    {
        CHECK(xsimd::add(2, 3) == 5);
        CHECK(xsimd::add(2.5, 0.5) == 3.0);
        CHECK(xsimd::sub(2, 3) == -1);
        CHECK(xsimd::sub(2.5, 0.5) == 2.0);
        CHECK(xsimd::mul(3, 4) == 12);
        CHECK(xsimd::mul(2.5, 2.0) == 5.0);
        CHECK(xsimd::div(12, 4) == 3);
        CHECK(xsimd::div(5.0, 2.0) == 2.5);
        CHECK(xsimd::mod(7, 3) == 1);
    }

    TEST_CASE("Avg")
    {
        CHECK(xsimd::avg(3, 4) == 3);
        CHECK(xsimd::avg(3u, 4u) == 3u);
        // rounding variants only differ for integral types
        CHECK(xsimd::avgr(3, 4) == 4);
        CHECK(xsimd::avgr(3u, 4u) == 4u);
    }

    TEST_CASE("SaddSsub")
    {
        constexpr auto int_max = std::numeric_limits<int>::max();
        constexpr auto int_low = std::numeric_limits<int>::lowest();
        CHECK(xsimd::sadd(int_max, 1) == int_max);
        CHECK(xsimd::sadd(int_low, -1) == int_low);
        CHECK(xsimd::sadd(1, 2) == 3);
    }

    TEST_CASE("SaddUnsignedSaturates")
    {
        constexpr auto u_max = std::numeric_limits<unsigned>::max();
        CHECK(xsimd::sadd(u_max, 1u) == u_max);
        CHECK(xsimd::sadd(1u, 2u) == 3u);

        CHECK(xsimd::ssub(u_max, u_max) == 0u);
        CHECK(xsimd::ssub(1u, 2u) == 0u); // saturates to lowest() == 0
    }

    TEST_CASE("IncrDecr")
    {
        CHECK(xsimd::incr(41) == 42);
        CHECK(xsimd::decr(43) == 42);
        CHECK(xsimd::incr_if(41, true) == 42);
        CHECK(xsimd::incr_if(41, false) == 41);
        CHECK(xsimd::decr_if(43, true) == 42);
        CHECK(xsimd::decr_if(43, false) == 43);
    }
}

TEST_SUITE("ScalarMath")
{
    TEST_CASE("Abs")
    {
        CHECK(xsimd::abs(-5) == 5);
        CHECK(xsimd::abs(5u) == 5u);
        CHECK(xsimd::abs(-2.5) == 2.5);
        CHECK(xsimd::abs(std::complex<double>(-3.0, -4.0)) == 5.0);
    }

    TEST_CASE("Clip")
    {
        CHECK(xsimd::clip(5, 1, 10) == 5);
        CHECK(xsimd::clip(-5, 1, 10) == 1);
        CHECK(xsimd::clip(50, 1, 10) == 10);
        CHECK(xsimd::clip(0.5, 0.0, 1.0) == 0.5);
        CHECK(xsimd::clip(-1.0, 0.0, 1.0) == 0.0);
    }

    TEST_CASE("SignAndSignbit")
    {
        CHECK(xsimd::sign(-3.0) == -1.0);
        CHECK(xsimd::sign(0.0) == 0.0);
        CHECK(xsimd::sign(3.0) == 1.0);
        CHECK(xsimd::signnz(0.0) == 1.0);
        CHECK(xsimd::signnz(-2.0) == -1.0);
        CHECK(xsimd::signbit(-1.5) == 1.0);
        CHECK(xsimd::signbit(1.5) == 0.0);

        auto s = xsimd::sign(std::complex<double>(-1.0, 2.0));
        CHECK(s.real() == -1.0);
        CHECK(s.imag() == 0.0);
    }

    TEST_CASE("MinMax")
    {
        CHECK(xsimd::min(3, 7) == 3);
        CHECK(xsimd::max(3, 7) == 7);
        CHECK(xsimd::min(1.25, 1.5) == 1.25);
        CHECK(xsimd::max(1.25, 1.5) == 1.5);
        // complex min/max use lexical ordering, like numpy
        auto cmin = xsimd::min(std::complex<double>(1.0, 5.0), std::complex<double>(2.0, 0.0));
        CHECK(cmin.real() == 1.0);
        auto cmax = xsimd::max(std::complex<double>(1.0, 5.0), std::complex<double>(2.0, 0.0));
        CHECK(cmax.real() == 2.0);
    }

    TEST_CASE("FmaFamily")
    {
        CHECK(xsimd::fma(2, 3, 4) == 10);
        CHECK(xsimd::fma(2.0, 3.0, 0.5) == 6.5);
        CHECK(xsimd::fms(2, 3, 4) == 2);
        CHECK(xsimd::fms(2.0, 3.0, 0.5) == 5.5);
        CHECK(xsimd::fnma(2, 3, 4) == -2);
        CHECK(xsimd::fnma(2.0, 3.0, 0.5) == -5.5);
        CHECK(xsimd::fnms(2, 3, 4) == -10);
        CHECK(xsimd::fnms(2.0, 3.0, 0.5) == -6.5);
    }

    TEST_CASE("ReciprocalRsqrt")
    {
        CHECK(xsimd::reciprocal(4.0f) == 0.25f);
        CHECK(xsimd::reciprocal(8.0) == 0.125);
        CHECK(xsimd::rsqrt(4.0) == doctest::Approx(0.5).epsilon(1e-12));
        CHECK(xsimd::rsqrt(4.0f) == doctest::Approx(0.5f).epsilon(1e-6));
    }

    TEST_CASE("Exp10")
    {
        CHECK(xsimd::exp10(2.0) == doctest::Approx(100.0).epsilon(1e-11));
        CHECK(xsimd::exp10(-1.0) == doctest::Approx(0.1).epsilon(1e-11));
        CHECK(xsimd::exp10(0.0f) == doctest::Approx(1.0f).epsilon(1e-6));
    }

    TEST_CASE("IntegerPow")
    {
        CHECK(xsimd::pow(2, 10) == 1024);
        CHECK((xsimd::pow)(2, 0) == 1);
        CHECK((xsimd::pow)(2.0, 10) == doctest::Approx(1024.0).epsilon(1e-12));
        CHECK((xsimd::pow)(2.0, -1) == doctest::Approx(0.5).epsilon(1e-12));
    }

    TEST_CASE("FlintEvenOdd")
    {
        CHECK(xsimd::is_flint(4.0));
        CHECK_FALSE(xsimd::is_flint(4.5));
        CHECK_FALSE(xsimd::is_flint(std::numeric_limits<double>::quiet_NaN()));
        CHECK(xsimd::is_even(4.0));
        CHECK_FALSE(xsimd::is_even(5.0));
        CHECK(xsimd::is_odd(5.0));
        CHECK_FALSE(xsimd::is_odd(4.0));
    }

    TEST_CASE("NearbyintAsInt")
    {
        CHECK(xsimd::nearbyint_as_int(2.4f) == 2);
        CHECK(xsimd::nearbyint_as_int(2.6) == 3);
        CHECK(xsimd::nearbyint_as_int(-2.5) == -2); // round half to even
    }

    TEST_CASE("Sincos")
    {
        auto sc = xsimd::sincos(0.0);
        CHECK(sc.first == 0.0);
        CHECK(sc.second == 1.0);
        auto scf = xsimd::sincos(1.0f);
        CHECK(scf.first == doctest::Approx(std::sin(1.0f)).epsilon(1e-6));
        CHECK(scf.second == doctest::Approx(std::cos(1.0f)).epsilon(1e-6));
    }
}

TEST_SUITE("ScalarBitwise")
{
    TEST_CASE("AndOrXorNot")
    {
        CHECK(xsimd::bitwise_and(0xF0F0, 0x0FF0) == 0x00F0);
        CHECK(xsimd::bitwise_or(0xF0F0, 0x000F) == 0xF0FF);
        CHECK(xsimd::bitwise_xor(0xF0F0, 0xFFFF) == 0x0F0F);
        CHECK(xsimd::bitwise_not(0x0F0Fu) == 0xFFFFF0F0u);
        CHECK(xsimd::bitwise_andnot(0xFF00, 0x00FF) == 0xFF00);

        // float / double overloads operate on the raw bit pattern
        float f = xsimd::bitwise_and(1.5f, 1.5f);
        CHECK(f == 1.5f);
        double d = xsimd::bitwise_or(0.0, 0.0);
        CHECK(d == 0.0);
        CHECK(xsimd::bitwise_not(xsimd::bitwise_not(2.5f)) == 2.5f);
    }

    TEST_CASE("Shifts")
    {
        CHECK(xsimd::bitwise_lshift(1, 4) == 16);
        CHECK(xsimd::bitwise_rshift(16, 4) == 1);
        CHECK((xsimd::bitwise_lshift<4>(1)) == 16);
        CHECK((xsimd::bitwise_rshift<4>(16)) == 1);
    }

    TEST_CASE("RotlRotr")
    {
        std::uint8_t x = 0b10000001;
        CHECK(xsimd::rotl(x, 1) == 0b00000011);
        CHECK(xsimd::rotr(x, 1) == 0b11000000);
        CHECK((xsimd::rotl<1>(x)) == 0b00000011);
        CHECK((xsimd::rotr<1>(x)) == 0b11000000);
    }

    TEST_CASE("BitwiseCast")
    {
        std::uint32_t bits = xsimd::bitwise_cast<std::uint32_t>(1.0f);
        CHECK(bits == 0x3F800000u);
        CHECK(xsimd::bitwise_cast<float>(0x3F800000u) == 1.0f);
    }
}

TEST_SUITE("ScalarPredicates")
{
    TEST_CASE("Comparisons")
    {
        CHECK(xsimd::eq(1, 1));
        CHECK_FALSE(xsimd::neq(1, 1));
        CHECK(xsimd::lt(1, 2));
        CHECK(xsimd::le(2, 2));
        CHECK(xsimd::gt(3, 2));
        CHECK(xsimd::ge(2, 2));
        CHECK(xsimd::all(true));
        CHECK(xsimd::any(true));
        CHECK(xsimd::none(false));
    }

    TEST_CASE("Classify")
    {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        constexpr double inf = std::numeric_limits<double>::infinity();
        CHECK(xsimd::isnan(nan));
        CHECK_FALSE(xsimd::isnan(1.0));
        CHECK(xsimd::isinf(inf));
        CHECK_FALSE(xsimd::isinf(1.0));
        CHECK(xsimd::isfinite(1.0));
        CHECK_FALSE(xsimd::isfinite(inf));
        CHECK_FALSE(xsimd::isfinite(nan));
    }
}

TEST_SUITE("ScalarMisc")
{
    TEST_CASE("SelectBatchCast")
    {
        CHECK(xsimd::select(true, 1, 2) == 1);
        CHECK(xsimd::select(false, 1, 2) == 2);
        CHECK((xsimd::batch_cast<double>(3)) == 3.0);
        CHECK(xsimd::batch_bool_cast<int>(true));
    }
}
