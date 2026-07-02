/**
 * UT1: std:: vs xsimd:: math functions comparison
 *
 * 1000 random inputs per function; compares relative error against a tolerance.
 * Types covered: float, double, complex<float>, complex<double>.
 *
 * Trig/exp/log/pow are expected to match within ~1e-6 (real) / ~1e-4..1e-5 (complex).
 * Rounding and abs/fabs/fmax/fmin must match exactly (bitwise).
 */

#include <gtest/gtest.h>
#include <xsimd/xsimd.hpp>

#include <complex>
#include <cstddef>
#include <random>
#include <vector>

namespace
{

    constexpr std::size_t N = 1000;

    template <typename T>
    std::vector<T> gen_real(T lo, T hi, std::size_t n)
    {
        std::mt19937 eng(42);
        std::uniform_real_distribution<T> d(lo, hi);
        std::vector<T> v(n);
        for (auto& x : v)
            x = d(eng);
        return v;
    }

    template <typename T>
    std::vector<std::complex<T>> gen_cx(T lo, T hi, std::size_t n)
    {
        std::mt19937 eng(42);
        std::uniform_real_distribution<T> d(lo, hi);
        std::vector<std::complex<T>> v(n);
        for (auto& c : v)
            c = { d(eng), d(eng) };
        return v;
    }

    // ---- unary real:  std:: vs xsimd:: with relative tolerance ----------------
    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_near_unary(const std::vector<T>& in, T tol_rel, T tol_abs,
                           ScalarFn sfn, VectorFn vfn)
    {
        using B = xsimd::batch<T>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            T so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(in[i + j]);
            xsimd::store_unaligned(vo, vfn(B::load_unaligned(&in[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_NEAR(so[j], vo[j], std::max(std::abs(so[j]) * tol_rel, tol_abs));
        }
    }

    // ---- binary real ----------------------------------------------------------
    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_near_binary(const std::vector<T>& a, const std::vector<T>& b,
                            T tol_rel, T tol_abs, ScalarFn sfn, VectorFn vfn)
    {
        using B = xsimd::batch<T>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            T so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(a[i + j], b[i + j]);
            xsimd::store_unaligned(vo, vfn(B::load_unaligned(&a[i]), B::load_unaligned(&b[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_NEAR(so[j], vo[j], std::max(std::abs(so[j]) * tol_rel, tol_abs));
        }
    }

    // ---- ternary real: fma ----------------------------------------------------
    template <typename T>
    void check_fma(const std::vector<T>& a, const std::vector<T>& b,
                   const std::vector<T>& c, T tol_rel, T tol_abs)
    {
        using B = xsimd::batch<T>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            T so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = std::fma(a[i + j], b[i + j], c[i + j]);
            xsimd::store_unaligned(vo, xsimd::fma(B::load_unaligned(&a[i]), B::load_unaligned(&b[i]), B::load_unaligned(&c[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_NEAR(so[j], vo[j], std::max(std::abs(so[j]) * tol_rel, tol_abs));
        }
    }

    // ---- exact-match helpers (rounding, abs, fmax, fmin) ----------------------
    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_eq_unary(const std::vector<T>& in, ScalarFn sfn, VectorFn vfn)
    {
        using B = xsimd::batch<T>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            T so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(in[i + j]);
            xsimd::store_unaligned(vo, vfn(B::load_unaligned(&in[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_EQ(so[j], vo[j]);
        }
    }

    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_eq_binary(const std::vector<T>& a, const std::vector<T>& b,
                          ScalarFn sfn, VectorFn vfn)
    {
        using B = xsimd::batch<T>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            T so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(a[i + j], b[i + j]);
            xsimd::store_unaligned(vo, vfn(B::load_unaligned(&a[i]), B::load_unaligned(&b[i])));
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_EQ(so[j], vo[j]);
        }
    }

    // ---- complex unary (compare real part only, as in original) ---------------
    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_near_cx_unary(const std::vector<std::complex<T>>& in,
                              T tol_rel, T tol_abs, ScalarFn sfn, VectorFn vfn)
    {
        using CT = std::complex<T>;
        using B = xsimd::batch<CT>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            CT so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(in[i + j]);
            vfn(B::load_unaligned(&in[i])).store_unaligned(vo);
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_NEAR(so[j].real(), vo[j].real(),
                            std::max(std::abs(so[j].real()) * tol_rel, tol_abs));
        }
    }

    template <typename T, typename ScalarFn, typename VectorFn>
    void expect_near_cx_binary(const std::vector<std::complex<T>>& a,
                               const std::vector<std::complex<T>>& b,
                               T tol_rel, T tol_abs, ScalarFn sfn, VectorFn vfn)
    {
        using CT = std::complex<T>;
        using B = xsimd::batch<CT>;
        constexpr std::size_t S = B::size;
        for (std::size_t i = 0; i < N; i += S)
        {
            CT so[S], vo[S];
            for (std::size_t j = 0; j < S; ++j)
                so[j] = sfn(a[i + j], b[i + j]);
            vfn(B::load_unaligned(&a[i]), B::load_unaligned(&b[i])).store_unaligned(vo);
            for (std::size_t j = 0; j < S; ++j)
                EXPECT_NEAR(so[j].real(), vo[j].real(),
                            std::max(std::abs(so[j].real()) * tol_rel, tol_abs));
        }
    }

} // namespace

// =========================================================================
// Real types: float, double
// =========================================================================

template <typename T>
class MathStdVsXsimdReal : public ::testing::Test
{
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MathStdVsXsimdReal, RealTypes);

TYPED_TEST(MathStdVsXsimdReal, Trig)
{
    using T = TypeParam;
    const T tol = static_cast<T>(1e-6), eps = static_cast<T>(0);
    auto v_sin = gen_real<T>(-6.28, 6.28, N);
    auto v_cos = gen_real<T>(-6.28, 6.28, N);
    auto v_tan = gen_real<T>(-1.5, 1.5, N);
    auto v_asin = gen_real<T>(-0.99, 0.99, N);
    auto v_acos = gen_real<T>(-0.99, 0.99, N);
    auto v_atan = gen_real<T>(-10., 10., N);
    auto v_ay = gen_real<T>(-10., 10., N);
    auto v_ax = gen_real<T>(-10., 10., N);

    expect_near_unary<T>(v_sin, tol, eps, [](T x)
                         { return std::sin(x); }, [](auto b)
                         { return xsimd::sin(b); });
    expect_near_unary<T>(v_cos, tol, eps, [](T x)
                         { return std::cos(x); }, [](auto b)
                         { return xsimd::cos(b); });
    expect_near_unary<T>(v_tan, tol, eps, [](T x)
                         { return std::tan(x); }, [](auto b)
                         { return xsimd::tan(b); });
    expect_near_unary<T>(v_asin, tol, eps, [](T x)
                         { return std::asin(x); }, [](auto b)
                         { return xsimd::asin(b); });
    expect_near_unary<T>(v_acos, tol, eps, [](T x)
                         { return std::acos(x); }, [](auto b)
                         { return xsimd::acos(b); });
    expect_near_unary<T>(v_atan, tol, eps, [](T x)
                         { return std::atan(x); }, [](auto b)
                         { return xsimd::atan(b); });
    expect_near_binary<T>(v_ay, v_ax, tol, eps, [](T y, T x)
                          { return std::atan2(y, x); }, [](auto by, auto bx)
                          { return xsimd::atan2(by, bx); });
}

TYPED_TEST(MathStdVsXsimdReal, Hyperbolic)
{
    using T = TypeParam;
    const T tol = static_cast<T>(1e-6), eps = static_cast<T>(0);
    auto v_sinh = gen_real<T>(-10., 10., N);
    auto v_cosh = gen_real<T>(-10., 10., N);
    auto v_tanh = gen_real<T>(-5., 5., N);
    auto v_asinh = gen_real<T>(-100., 100., N);
    auto v_acosh = gen_real<T>(1.001, 100., N);
    auto v_atanh = gen_real<T>(-0.99, 0.99, N);

    expect_near_unary<T>(v_sinh, tol, eps, [](T x)
                         { return std::sinh(x); }, [](auto b)
                         { return xsimd::sinh(b); });
    expect_near_unary<T>(v_cosh, tol, eps, [](T x)
                         { return std::cosh(x); }, [](auto b)
                         { return xsimd::cosh(b); });
    expect_near_unary<T>(v_tanh, tol, eps, [](T x)
                         { return std::tanh(x); }, [](auto b)
                         { return xsimd::tanh(b); });
    expect_near_unary<T>(v_asinh, tol, eps, [](T x)
                         { return std::asinh(x); }, [](auto b)
                         { return xsimd::asinh(b); });
    expect_near_unary<T>(v_acosh, tol, eps, [](T x)
                         { return std::acosh(x); }, [](auto b)
                         { return xsimd::acosh(b); });
    expect_near_unary<T>(v_atanh, tol, eps, [](T x)
                         { return std::atanh(x); }, [](auto b)
                         { return xsimd::atanh(b); });
}

TYPED_TEST(MathStdVsXsimdReal, ExpLog)
{
    using T = TypeParam;
    const T tol = static_cast<T>(1e-6), eps = static_cast<T>(0);
    auto v_exp = gen_real<T>(-10., 10., N);
    auto v_exp2 = gen_real<T>(-10., 10., N);
    auto v_expm1 = gen_real<T>(-10., 10., N);
    auto v_log = gen_real<T>(0.001, 1000., N);
    auto v_log2 = gen_real<T>(0.001, 1000., N);
    auto v_log10 = gen_real<T>(0.001, 1000., N);
    auto v_log1p = gen_real<T>(-0.99, 100., N);

    expect_near_unary<T>(v_exp, tol, eps, [](T x)
                         { return std::exp(x); }, [](auto b)
                         { return xsimd::exp(b); });
    expect_near_unary<T>(v_exp2, tol, eps, [](T x)
                         { return std::exp2(x); }, [](auto b)
                         { return xsimd::exp2(b); });
    expect_near_unary<T>(v_expm1, tol, eps, [](T x)
                         { return std::expm1(x); }, [](auto b)
                         { return xsimd::expm1(b); });
    expect_near_unary<T>(v_log, tol, eps, [](T x)
                         { return std::log(x); }, [](auto b)
                         { return xsimd::log(b); });
    expect_near_unary<T>(v_log2, tol, eps, [](T x)
                         { return std::log2(x); }, [](auto b)
                         { return xsimd::log2(b); });
    expect_near_unary<T>(v_log10, tol, eps, [](T x)
                         { return std::log10(x); }, [](auto b)
                         { return xsimd::log10(b); });
    expect_near_unary<T>(v_log1p, tol, eps, [](T x)
                         { return std::log1p(x); }, [](auto b)
                         { return xsimd::log1p(b); });
}

TYPED_TEST(MathStdVsXsimdReal, PowRoot)
{
    using T = TypeParam;
    // cbrt/pow need a looser tolerance for float (2e-5), but 1e-6 is fine for double.
    const T tol = static_cast<T>(1e-6);
    const T tol_loose = sizeof(T) == 4 ? static_cast<T>(2e-5) : static_cast<T>(1e-6);
    auto v_sqrt = gen_real<T>(0.001, 100., N);
    auto v_cbrt = gen_real<T>(-100., 100., N);
    auto v_powa = gen_real<T>(0.01, 10., N);
    auto v_powb = gen_real<T>(-2., 3., N);
    auto v_hya = gen_real<T>(-100., 100., N);
    auto v_hyb = gen_real<T>(-100., 100., N);

    expect_near_unary<T>(v_sqrt, tol, static_cast<T>(0), [](T x)
                         { return std::sqrt(x); }, [](auto b)
                         { return xsimd::sqrt(b); });
    expect_near_unary<T>(v_cbrt, tol_loose, static_cast<T>(0), [](T x)
                         { return std::cbrt(x); }, [](auto b)
                         { return xsimd::cbrt(b); });
    expect_near_binary<T>(v_powa, v_powb, tol_loose, static_cast<T>(0), [](T x, T y)
                          { return std::pow(x, y); }, [](auto bx, auto by)
                          { return xsimd::pow(bx, by); });
    expect_near_binary<T>(v_hya, v_hyb, tol, static_cast<T>(0), [](T x, T y)
                          { return std::hypot(x, y); }, [](auto bx, auto by)
                          { return xsimd::hypot(bx, by); });
}

TYPED_TEST(MathStdVsXsimdReal, Rounding)
{
    using T = TypeParam;
    auto v = gen_real<T>(-100., 100., N);
    expect_eq_unary<T>(v, [](T x)
                       { return std::ceil(x); }, [](auto b)
                       { return xsimd::ceil(b); });
    expect_eq_unary<T>(v, [](T x)
                       { return std::floor(x); }, [](auto b)
                       { return xsimd::floor(b); });
    expect_eq_unary<T>(v, [](T x)
                       { return std::trunc(x); }, [](auto b)
                       { return xsimd::trunc(b); });
    expect_eq_unary<T>(v, [](T x)
                       { return std::round(x); }, [](auto b)
                       { return xsimd::round(b); });
}

TYPED_TEST(MathStdVsXsimdReal, FmodFma)
{
    using T = TypeParam;
    const T tol = static_cast<T>(1e-5), eps = static_cast<T>(1e-6);
    auto v_a = gen_real<T>(0.1, 10., N);
    auto v_b = gen_real<T>(0.1, 10., N);
    auto v_c = gen_real<T>(-10., 10., N);

    expect_near_binary<T>(v_a, v_b, tol, eps, [](T x, T y)
                          { return std::fmod(x, y); }, [](auto bx, auto by)
                          { return xsimd::fmod(bx, by); });
    check_fma<T>(v_a, v_b, v_c, tol, eps);
}

TYPED_TEST(MathStdVsXsimdReal, AbsFabsFmaxFmin)
{
    using T = TypeParam;
    auto v = gen_real<T>(-1000., 1000., N);
    auto v_a = gen_real<T>(-100., 100., N);
    auto v_b = gen_real<T>(-100., 100., N);
    expect_eq_unary<T>(v, [](T x)
                       { return std::abs(x); }, [](auto b)
                       { return xsimd::abs(b); });
    expect_eq_unary<T>(v, [](T x)
                       { return std::fabs(x); }, [](auto b)
                       { return xsimd::fabs(b); });
    expect_eq_binary<T>(v_a, v_b, [](T x, T y)
                        { return std::fmax(x, y); }, [](auto bx, auto by)
                        { return xsimd::fmax(bx, by); });
    expect_eq_binary<T>(v_a, v_b, [](T x, T y)
                        { return std::fmin(x, y); }, [](auto bx, auto by)
                        { return xsimd::fmin(bx, by); });
}

// =========================================================================
// Complex types: complex<float>, complex<double>
// =========================================================================

template <typename T>
class MathStdVsXsimdComplex : public ::testing::Test
{
};
using CxTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MathStdVsXsimdComplex, CxTypes);

TYPED_TEST(MathStdVsXsimdComplex, TrigHyperbolic)
{
    using CT = TypeParam;
    using T = typename CT::value_type;
    // float -> 1e-4 rel / 1e-5 abs ; double -> 1e-5 rel / 1e-10 abs
    const T tol_rel = sizeof(T) == 4 ? static_cast<T>(1e-4) : static_cast<T>(1e-5);
    const T tol_abs = sizeof(T) == 4 ? static_cast<T>(1e-5) : static_cast<T>(1e-10);

    auto v_sin = gen_cx<T>(-6.28, 6.28, N);
    auto v_cos = gen_cx<T>(-6.28, 6.28, N);
    auto v_tan = gen_cx<T>(-1.5, 1.5, N);
    auto v_asin = gen_cx<T>(-0.99, 0.99, N);
    auto v_acos = gen_cx<T>(-0.99, 0.99, N);
    auto v_atan = gen_cx<T>(-5., 5., N);
    auto v_sinh = gen_cx<T>(-5., 5., N);
    auto v_cosh = gen_cx<T>(-5., 5., N);
    auto v_tanh = gen_cx<T>(-5., 5., N);
    auto v_ash = gen_cx<T>(-10., 10., N);
    auto v_ach = gen_cx<T>(1.01, 10., N);
    auto v_ath = gen_cx<T>(-0.99, 0.99, N);

    expect_near_cx_unary<T>(v_sin, tol_rel, tol_abs, [](CT x)
                            { return std::sin(x); }, [](auto b)
                            { return xsimd::sin(b); });
    expect_near_cx_unary<T>(v_cos, tol_rel, tol_abs, [](CT x)
                            { return std::cos(x); }, [](auto b)
                            { return xsimd::cos(b); });
    expect_near_cx_unary<T>(v_tan, tol_rel, tol_abs, [](CT x)
                            { return std::tan(x); }, [](auto b)
                            { return xsimd::tan(b); });
    expect_near_cx_unary<T>(v_asin, tol_rel, tol_abs, [](CT x)
                            { return std::asin(x); }, [](auto b)
                            { return xsimd::asin(b); });
    expect_near_cx_unary<T>(v_acos, tol_rel, tol_abs, [](CT x)
                            { return std::acos(x); }, [](auto b)
                            { return xsimd::acos(b); });
    expect_near_cx_unary<T>(v_atan, tol_rel, tol_abs, [](CT x)
                            { return std::atan(x); }, [](auto b)
                            { return xsimd::atan(b); });
    expect_near_cx_unary<T>(v_sinh, tol_rel, tol_abs, [](CT x)
                            { return std::sinh(x); }, [](auto b)
                            { return xsimd::sinh(b); });
    expect_near_cx_unary<T>(v_cosh, tol_rel, tol_abs, [](CT x)
                            { return std::cosh(x); }, [](auto b)
                            { return xsimd::cosh(b); });
    expect_near_cx_unary<T>(v_tanh, tol_rel, tol_abs, [](CT x)
                            { return std::tanh(x); }, [](auto b)
                            { return xsimd::tanh(b); });
    expect_near_cx_unary<T>(v_ash, tol_rel, tol_abs, [](CT x)
                            { return std::asinh(x); }, [](auto b)
                            { return xsimd::asinh(b); });
    expect_near_cx_unary<T>(v_ach, tol_rel, tol_abs, [](CT x)
                            { return std::acosh(x); }, [](auto b)
                            { return xsimd::acosh(b); });
    expect_near_cx_unary<T>(v_ath, tol_rel, tol_abs, [](CT x)
                            { return std::atanh(x); }, [](auto b)
                            { return xsimd::atanh(b); });
}

TYPED_TEST(MathStdVsXsimdComplex, ExpLogSqrtPow)
{
    using CT = TypeParam;
    using T = typename CT::value_type;
    const T tol_rel = sizeof(T) == 4 ? static_cast<T>(1e-4) : static_cast<T>(1e-5);
    const T tol_abs = sizeof(T) == 4 ? static_cast<T>(1e-5) : static_cast<T>(1e-10);

    auto v_exp = gen_cx<T>(-5., 5., N);
    auto v_log = gen_cx<T>(0.01, 100., N);
    auto v_sqrt = gen_cx<T>(0.01, 100., N);
    auto v_powa = gen_cx<T>(0.01, 10., N);
    auto v_powb = gen_cx<T>(-2., 2., N);

    expect_near_cx_unary<T>(v_exp, tol_rel, tol_abs, [](CT x)
                            { return std::exp(x); }, [](auto b)
                            { return xsimd::exp(b); });
    expect_near_cx_unary<T>(v_log, tol_rel, tol_abs, [](CT x)
                            { return std::log(x); }, [](auto b)
                            { return xsimd::log(b); });
    expect_near_cx_unary<T>(v_sqrt, tol_rel, tol_abs, [](CT x)
                            { return std::sqrt(x); }, [](auto b)
                            { return xsimd::sqrt(b); });
    expect_near_cx_binary<T>(v_powa, v_powb, tol_rel, tol_abs, [](CT x, CT y)
                             { return std::pow(x, y); }, [](auto bx, auto by)
                             { return xsimd::pow(bx, by); });
}

TYPED_TEST(MathStdVsXsimdComplex, Accessors)
{
    using CT = TypeParam;
    using T = typename CT::value_type;
    using B = xsimd::batch<CT>;
    constexpr std::size_t S = B::size;
    auto v = gen_cx<T>(-10., 10., N);
    const T tol = static_cast<T>(1e-6);

    for (std::size_t i = 0; i < N; i += S)
    {
        T rso[S], iso[S], rvo[S], ivo[S];
        B b = B::load_unaligned(&v[i]);

        for (std::size_t j = 0; j < S; ++j)
            rso[j] = std::real(v[i + j]);
        xsimd::store_unaligned(rvo, xsimd::real(b));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_EQ(rso[j], rvo[j]);

        for (std::size_t j = 0; j < S; ++j)
            iso[j] = std::imag(v[i + j]);
        xsimd::store_unaligned(ivo, xsimd::imag(b));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_EQ(iso[j], ivo[j]);

        for (std::size_t j = 0; j < S; ++j)
            rso[j] = std::abs(v[i + j]);
        xsimd::store_unaligned(rvo, xsimd::abs(b));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_NEAR(rso[j], rvo[j], std::abs(rso[j]) * tol);

        for (std::size_t j = 0; j < S; ++j)
            rso[j] = std::arg(v[i + j]);
        xsimd::store_unaligned(rvo, xsimd::arg(b));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_NEAR(rso[j], rvo[j], std::abs(rso[j]) * tol);

        for (std::size_t j = 0; j < S; ++j)
            rso[j] = std::norm(v[i + j]);
        xsimd::store_unaligned(rvo, xsimd::norm(b));
        for (std::size_t j = 0; j < S; ++j)
            EXPECT_NEAR(rso[j], rvo[j], std::abs(rso[j]) * tol);

        for (std::size_t j = 0; j < S; ++j)
        {
            CT c = std::conj(v[i + j]);
            rso[j] = c.real();
            iso[j] = c.imag();
        }
        {
            auto cb = xsimd::conj(b);
            xsimd::store_unaligned(rvo, xsimd::real(cb));
            xsimd::store_unaligned(ivo, xsimd::imag(cb));
        }
        for (std::size_t j = 0; j < S; ++j)
        {
            EXPECT_EQ(rso[j], rvo[j]);
            EXPECT_EQ(iso[j], ivo[j]);
        }
    }
}
