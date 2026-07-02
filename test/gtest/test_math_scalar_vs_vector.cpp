/**
 * UT2: xsimd scalar vs vector bitwise comparison
 *
 * Compares xsimd scalar (std:: wrapper) against vectorized results.
 * Only tests functions where scalar and vector results are bitwise identical:
 *   rounding (ceil/floor/trunc/round), fmod, fma, abs, fabs, fmax, fmin, sqrt.
 *
 * Trig/exp/log/pow use different algorithm paths for scalar vs vector, so
 * they are NOT expected to be bitwise identical (covered by UT1 instead).
 *
 * Compiled for AVX2 and AVX512.
 */

#include <gtest/gtest.h>
#include <xsimd/xsimd.hpp>

#include <random>
#include <vector>

// ARCH_SUFFIX is defined by CMake (e.g. AVX2, AVX512) to produce
// distinct test-suite names so that the two arch variants don't collide.
#ifndef ARCH_SUFFIX
#define ARCH_SUFFIX Default
#endif

// Token-pasting helpers to build the test-suite name.
// The double-indirection (XS_SV_CONCAT2 -> XS_SV_CONCAT1) is required so that
// ARCH_SUFFIX is fully expanded before the ## operator is applied.
#define XS_SV_CONCAT1(a, b) a##_##b
#define XS_SV_CONCAT2(a, b) XS_SV_CONCAT1(a, b)
#define XS_SV_SUITE        XS_SV_CONCAT2(ScalarVsVector, ARCH_SUFFIX)

namespace
{

    constexpr std::size_t N = 1000;

    std::mt19937& rng()
    {
        static std::mt19937 eng(42);
        return eng;
    }

    std::vector<float> gen_real(float lo, float hi, std::size_t n)
    {
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v(n);
        for (auto& x : v)
            x = d(rng());
        return v;
    }

    // Compare scalar function vs vectorized function, element-wise, bitwise.
    //   sfn : scalar callable   float -> float
    //   vfn : vector  callable  batch  -> batch
    template <typename ScalarFn, typename VectorFn>
    void expect_eq_vec(ScalarFn sfn, VectorFn vfn)
    {
        constexpr std::size_t S = xsimd::batch<float>::size;
        auto in = gen_real(-100.0f, 100.0f, N);
        for (std::size_t i = 0; i < N; i += S)
        {
            float so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(in[i + j]);
            xsimd::store_unaligned(vo, vfn(xsimd::load_unaligned(&in[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_EQ(so[j], vo[j]);
        }
    }

    // Compare scalar binary function vs vectorized binary function, bitwise.
    template <typename ScalarFn, typename VectorFn>
    void expect_eq_vec2(ScalarFn sfn, VectorFn vfn)
    {
        constexpr std::size_t S = xsimd::batch<float>::size;
        auto a = gen_real(0.1f, 100.0f, N);
        auto b = gen_real(0.1f, 100.0f, N);
        for (std::size_t i = 0; i < N; i += S)
        {
            float so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(a[i + j], b[i + j]);
            xsimd::store_unaligned(vo, vfn(xsimd::load_unaligned(&a[i]), xsimd::load_unaligned(&b[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_EQ(so[j], vo[j]);
        }
    }

} // namespace

// ========== Unary functions ==========

TEST(XS_SV_SUITE, Ceil)
{
    expect_eq_vec([](float x)
                  { return xsimd::ceil(x); }, [](auto b)
                  { return xsimd::ceil(b); });
}
TEST(XS_SV_SUITE, Floor)
{
    expect_eq_vec([](float x)
                  { return xsimd::floor(x); }, [](auto b)
                  { return xsimd::floor(b); });
}
TEST(XS_SV_SUITE, Trunc)
{
    expect_eq_vec([](float x)
                  { return xsimd::trunc(x); }, [](auto b)
                  { return xsimd::trunc(b); });
}
TEST(XS_SV_SUITE, Round)
{
    expect_eq_vec([](float x)
                  { return xsimd::round(x); }, [](auto b)
                  { return xsimd::round(b); });
}

TEST(XS_SV_SUITE, Abs)
{
    expect_eq_vec([](float x)
                  { return xsimd::abs(x); }, [](auto b)
                  { return xsimd::abs(b); });
}
TEST(XS_SV_SUITE, Fabs)
{
    expect_eq_vec([](float x)
                  { return xsimd::fabs(x); }, [](auto b)
                  { return xsimd::fabs(b); });
}

// Sqrt needs a non-negative input range.
TEST(XS_SV_SUITE, Sqrt)
{
    constexpr std::size_t S = xsimd::batch<float>::size;
    auto in = gen_real(0.001f, 10000.0f, N);
    for (std::size_t i = 0; i < N; i += S)
    {
        float so[S], vo[S];
        for (std::size_t j = 0; j < S; ++j)
            so[j] = xsimd::sqrt(in[i + j]);
        xsimd::store_unaligned(vo, xsimd::sqrt(xsimd::load_unaligned(&in[i])));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_EQ(so[j], vo[j]);
    }
}

// ========== Binary functions ==========

TEST(XS_SV_SUITE, Fmod)
{
    expect_eq_vec2([](float x, float y)
                   { return xsimd::fmod(x, y); },
                   [](auto bx, auto by)
                   { return xsimd::fmod(bx, by); });
}
TEST(XS_SV_SUITE, Fmax)
{
    expect_eq_vec2([](float x, float y)
                   { return xsimd::fmax(x, y); },
                   [](auto bx, auto by)
                   { return xsimd::fmax(bx, by); });
}
TEST(XS_SV_SUITE, Fmin)
{
    expect_eq_vec2([](float x, float y)
                   { return xsimd::fmin(x, y); },
                   [](auto bx, auto by)
                   { return xsimd::fmin(bx, by); });
}

// ========== Ternary function: fma ==========

TEST(XS_SV_SUITE, Fma)
{
    constexpr std::size_t S = xsimd::batch<float>::size;
    auto a = gen_real(-100.0f, 100.0f, N);
    auto b = gen_real(-100.0f, 100.0f, N);
    auto c = gen_real(-100.0f, 100.0f, N);
    for (std::size_t i = 0; i < N; i += S)
    {
        float so[S], vo[S];
        for (std::size_t j = 0; j < S; ++j)
            so[j] = xsimd::fma(a[i + j], b[i + j], c[i + j]);
        xsimd::store_unaligned(vo, xsimd::fma(xsimd::load_unaligned(&a[i]), xsimd::load_unaligned(&b[i]), xsimd::load_unaligned(&c[i])));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_EQ(so[j], vo[j]);
    }
}
