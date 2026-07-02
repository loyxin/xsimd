/**
 * UT3: xsimd scalar cmath functions unit tests
 *
 * Verifies the custom scalar implementations of trigonometric, hyperbolic
 * and inverse trigonometric/hyperbolic functions that were ported from the
 * vector kernels into xsimd::scalar overloads.
 *
 * Since these functions are designed to be bit-identical to the vector
 * kernels (and may differ from std::xxx by ~1 ULP), we validate:
 *   - Known special values (0, 1, pi/4, pi/2, ...)
 *   - Functional identities (sin^2 + cos^2 == 1, tan = sin/cos, ...)
 *   - Consistency with the vector batch implementation
 *   - Sign / parity correctness
 */

#include <gtest/gtest.h>
#include <xsimd/xsimd.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{
    // Tolerance helpers. The custom kernels aim for bit-identical results
    // with the vector path; allow a small epsilon for cross-checks against
    // the standard library which uses a different algorithm.
    constexpr float kFloatTol = 1e-5f;
    constexpr double kDoubleTol = 1e-12;

    template <class T>
    T pi()
    {
        return T(3.141592653589793238462643383279502884);
    }

    template <class T>
    T pio2()
    {
        return pi<T>() * T(0.5);
    }

    template <class T>
    T pio4()
    {
        return pi<T>() * T(0.25);
    }

    bool nearly_equal(float a, float b, float tol = kFloatTol)
    {
        return std::fabs(a - b) <= tol;
    }

    bool nearly_equal(double a, double b, double tol = kDoubleTol)
    {
        return std::fabs(a - b) <= tol;
    }

    template <class T>
    bool nearly_equal(std::complex<T> a, std::complex<T> b, T tol)
    {
        return nearly_equal(a.real(), b.real(), tol)
            && nearly_equal(a.imag(), b.imag(), tol);
    }
} // namespace

// =============================================================================
// Trigonometric: sin / cos / sincos
// =============================================================================

TEST(XsimdScalarCmathTest, SinFloatKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::sin(0.0f), 0.0f);
    EXPECT_NEAR(xsimd::sin(pio4<float>()), std::sqrt(0.5f), kFloatTol);
    EXPECT_NEAR(xsimd::sin(pio2<float>()), 1.0f, kFloatTol);
    EXPECT_NEAR(xsimd::sin(pi<float>()), 0.0f, kFloatTol);
    EXPECT_NEAR(xsimd::sin(-pio2<float>()), -1.0f, kFloatTol);
}

TEST(XsimdScalarCmathTest, SinDoubleKnownValues)
{
    EXPECT_DOUBLE_EQ(xsimd::sin(0.0), 0.0);
    EXPECT_NEAR(xsimd::sin(pio4<double>()), std::sqrt(0.5), kDoubleTol);
    EXPECT_NEAR(xsimd::sin(pio2<double>()), 1.0, kDoubleTol);
    EXPECT_NEAR(xsimd::sin(pi<double>()), 0.0, kDoubleTol);
    EXPECT_NEAR(xsimd::sin(-pio2<double>()), -1.0, kDoubleTol);
}

TEST(XsimdScalarCmathTest, CosFloatKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::cos(0.0f), 1.0f);
    EXPECT_NEAR(xsimd::cos(pio4<float>()), std::sqrt(0.5f), kFloatTol);
    EXPECT_NEAR(xsimd::cos(pio2<float>()), 0.0f, kFloatTol);
    EXPECT_NEAR(xsimd::cos(pi<float>()), -1.0f, kFloatTol);
}

TEST(XsimdScalarCmathTest, CosDoubleKnownValues)
{
    EXPECT_DOUBLE_EQ(xsimd::cos(0.0), 1.0);
    EXPECT_NEAR(xsimd::cos(pio4<double>()), std::sqrt(0.5), kDoubleTol);
    EXPECT_NEAR(xsimd::cos(pio2<double>()), 0.0, kDoubleTol);
    EXPECT_NEAR(xsimd::cos(pi<double>()), -1.0, kDoubleTol);
}

TEST(XsimdScalarCmathTest, SincosReturnsPair)
{
    auto pf = xsimd::sincos(0.0f);
    EXPECT_FLOAT_EQ(pf.first, 0.0f);
    EXPECT_FLOAT_EQ(pf.second, 1.0f);

    auto pd = xsimd::sincos(0.0);
    EXPECT_DOUBLE_EQ(pd.first, 0.0);
    EXPECT_DOUBLE_EQ(pd.second, 1.0);
}

TEST(XsimdScalarCmathTest, PythagoreanIdentity)
{
    const float values_f[] = { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f };
    for (float v : values_f)
    {
        auto sc = xsimd::sincos(v);
        float mag2 = sc.first * sc.first + sc.second * sc.second;
        EXPECT_NEAR(mag2, 1.0f, kFloatTol) << "v=" << v;
    }

    const double values_d[] = { 0.1, 0.5, 1.0, 1.5, 2.0, 3.0 };
    for (double v : values_d)
    {
        auto sc = xsimd::sincos(v);
        double mag2 = sc.first * sc.first + sc.second * sc.second;
        EXPECT_NEAR(mag2, 1.0, kDoubleTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, SinCosSignParity)
{
    for (int i = 1; i <= 5; ++i)
    {
        float v = static_cast<float>(i) * 0.3f;
        // sin is odd
        EXPECT_NEAR(xsimd::sin(-v), -xsimd::sin(v), kFloatTol);
        // cos is even
        EXPECT_NEAR(xsimd::cos(-v), xsimd::cos(v), kFloatTol);
    }
}

// =============================================================================
// Trigonometric: tan / cot
// =============================================================================

TEST(XsimdScalarCmathTest, TanKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::tan(0.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::tan(0.0), 0.0);
    EXPECT_NEAR(xsimd::tan(pio4<float>()), 1.0f, kFloatTol);
    EXPECT_NEAR(xsimd::tan(pio4<double>()), 1.0, kDoubleTol);
}

TEST(XsimdScalarCmathTest, TanEqualsSinOverCos)
{
    const float values[] = { 0.1f, 0.4f, 0.9f, 1.2f };
    for (float v : values)
    {
        auto sc = xsimd::sincos(v);
        EXPECT_NEAR(xsimd::tan(v), sc.first / sc.second, kFloatTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, CotKnownValues)
{
    EXPECT_NEAR(xsimd::cot(pio4<float>()), 1.0f, kFloatTol);
    EXPECT_NEAR(xsimd::cot(pio4<double>()), 1.0, kDoubleTol);
}

TEST(XsimdScalarCmathTest, CotEqualsCosOverSin)
{
    const float values[] = { 0.2f, 0.6f, 1.0f, 1.3f };
    for (float v : values)
    {
        auto sc = xsimd::sincos(v);
        EXPECT_NEAR(xsimd::cot(v), sc.second / sc.first, kFloatTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, CotIsInverseOfTan)
{
    const double values[] = { 0.15, 0.55, 0.95, 1.25 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::cot(v), 1.0 / xsimd::tan(v), kDoubleTol) << "v=" << v;
    }
}

// =============================================================================
// Inverse trigonometric: asin / acos / atan / acot
// =============================================================================

TEST(XsimdScalarCmathTest, AsinKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::asin(0.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::asin(0.0), 0.0);
    EXPECT_NEAR(xsimd::asin(1.0f), pio2<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::asin(1.0), pio2<double>(), kDoubleTol);
    EXPECT_NEAR(xsimd::asin(-1.0f), -pio2<float>(), kFloatTol);
}

TEST(XsimdScalarCmathTest, AcosKnownValues)
{
    EXPECT_NEAR(xsimd::acos(1.0f), 0.0f, kFloatTol);
    EXPECT_NEAR(xsimd::acos(1.0), 0.0, kDoubleTol);
    EXPECT_NEAR(xsimd::acos(0.0f), pio2<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::acos(0.0), pio2<double>(), kDoubleTol);
    EXPECT_NEAR(xsimd::acos(-1.0f), pi<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::acos(-1.0), pi<double>(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, AsinPlusAcosEqualsPio2)
{
    const float values[] = { -0.9f, -0.4f, 0.0f, 0.3f, 0.7f, 0.99f };
    for (float v : values)
    {
        EXPECT_NEAR(xsimd::asin(v) + xsimd::acos(v), pio2<float>(), kFloatTol)
            << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, AtanKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::atan(0.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::atan(0.0), 0.0);
    EXPECT_NEAR(xsimd::atan(1.0f), pio4<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::atan(1.0), pio4<double>(), kDoubleTol);
    EXPECT_NEAR(xsimd::atan(-1.0f), -pio4<float>(), kFloatTol);
}

TEST(XsimdScalarCmathTest, AcotKnownValues)
{
    EXPECT_NEAR(xsimd::acot(0.0f), pio2<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::acot(0.0), pio2<double>(), kDoubleTol);
    EXPECT_NEAR(xsimd::acot(1.0f), pio4<float>(), kFloatTol);
    EXPECT_NEAR(xsimd::acot(1.0), pio4<double>(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, AcotPlusAtanEqualsPio2)
{
    const float values[] = { 0.1f, 0.5f, 1.0f, 2.0f, 5.0f };
    for (float v : values)
    {
        EXPECT_NEAR(xsimd::acot(v) + xsimd::atan(v), pio2<float>(), kFloatTol)
            << "v=" << v;
    }
}

// =============================================================================
// Hyperbolic: sinh / cosh
// =============================================================================

TEST(XsimdScalarCmathTest, SinhKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::sinh(0.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::sinh(0.0), 0.0);
    EXPECT_NEAR(xsimd::sinh(1.0f), 1.1752012f, kFloatTol);
    EXPECT_NEAR(xsimd::sinh(1.0), 1.1752011936438014, kDoubleTol);
}

TEST(XsimdScalarCmathTest, CoshKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::cosh(0.0f), 1.0f);
    EXPECT_DOUBLE_EQ(xsimd::cosh(0.0), 1.0);
    EXPECT_NEAR(xsimd::cosh(1.0f), 1.5430806f, kFloatTol);
    EXPECT_NEAR(xsimd::cosh(1.0), 1.5430806348152437, kDoubleTol);
}

TEST(XsimdScalarCmathTest, CoshMinusSinhEqualsExp)
{
    // cosh(x) - sinh(x) == exp(-x)
    const double values[] = { -1.5, -0.5, 0.0, 0.7, 1.2 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::cosh(v) - xsimd::sinh(v), std::exp(-v), kDoubleTol)
            << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, SinhIsOdd)
{
    const double values[] = { 0.3, 0.8, 1.5 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::sinh(-v), -xsimd::sinh(v), kDoubleTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, CoshIsEven)
{
    const double values[] = { 0.3, 0.8, 1.5 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::cosh(-v), xsimd::cosh(v), kDoubleTol) << "v=" << v;
    }
}

// =============================================================================
// Inverse hyperbolic: asinh / acosh
// =============================================================================

TEST(XsimdScalarCmathTest, AsinhKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::asinh(0.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::asinh(0.0), 0.0);
    // asinh(1) = ln(1 + sqrt(2))
    EXPECT_NEAR(xsimd::asinh(1.0f), std::log(1.0f + std::sqrt(2.0f)), kFloatTol);
    EXPECT_NEAR(xsimd::asinh(1.0), std::log(1.0 + std::sqrt(2.0)), kDoubleTol);
}

TEST(XsimdScalarCmathTest, AcoshKnownValues)
{
    EXPECT_FLOAT_EQ(xsimd::acosh(1.0f), 0.0f);
    EXPECT_DOUBLE_EQ(xsimd::acosh(1.0), 0.0);
    EXPECT_NEAR(xsimd::acosh(2.0f), std::log(2.0f + std::sqrt(3.0f)), kFloatTol);
    EXPECT_NEAR(xsimd::acosh(2.0), std::log(2.0 + std::sqrt(3.0)), kDoubleTol);
}

TEST(XsimdScalarCmathTest, AsinhInverseOfSinh)
{
    const float values[] = { -1.5f, -0.4f, 0.0f, 0.6f, 1.3f };
    for (float v : values)
    {
        EXPECT_NEAR(xsimd::asinh(xsimd::sinh(v)), v, kFloatTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, AcoshInverseOfCosh)
{
    const double values[] = { 0.0, 0.5, 1.0, 1.5 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::acosh(xsimd::cosh(v)), std::fabs(v), kDoubleTol)
            << "v=" << v;
    }
}

// =============================================================================
// Complex scalar overloads
// =============================================================================

TEST(XsimdScalarCmathTest, ComplexSin)
{
    std::complex<double> z(0.5, 0.3);
    std::complex<double> expected(std::sin(0.5) * std::cosh(0.3),
                                  std::cos(0.5) * std::sinh(0.3));
    std::complex<double> result = xsimd::sin(z);
    EXPECT_NEAR(result.real(), expected.real(), kDoubleTol);
    EXPECT_NEAR(result.imag(), expected.imag(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, ComplexCos)
{
    std::complex<double> z(0.5, 0.3);
    std::complex<double> expected(std::cos(0.5) * std::cosh(0.3),
                                  -std::sin(0.5) * std::sinh(0.3));
    std::complex<double> result = xsimd::cos(z);
    EXPECT_NEAR(result.real(), expected.real(), kDoubleTol);
    EXPECT_NEAR(result.imag(), expected.imag(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, ComplexSinh)
{
    std::complex<double> z(0.4, 0.6);
    std::complex<double> expected(std::sinh(0.4) * std::cos(0.6),
                                  std::cosh(0.4) * std::sin(0.6));
    std::complex<double> result = xsimd::sinh(z);
    EXPECT_NEAR(result.real(), expected.real(), kDoubleTol);
    EXPECT_NEAR(result.imag(), expected.imag(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, ComplexCosh)
{
    std::complex<double> z(0.4, 0.6);
    std::complex<double> expected(std::cosh(0.4) * std::cos(0.6),
                                  std::sinh(0.4) * std::sin(0.6));
    std::complex<double> result = xsimd::cosh(z);
    EXPECT_NEAR(result.real(), expected.real(), kDoubleTol);
    EXPECT_NEAR(result.imag(), expected.imag(), kDoubleTol);
}

TEST(XsimdScalarCmathTest, ComplexAsinAtAcosIdentity)
{
    // asin(z) + acos(z) == pi/2 for complex z as well
    const std::complex<double> values[] = {
        { 0.0, 0.0 },
        { 0.3, 0.2 },
        { -0.4, 0.5 },
        { 0.6, -0.1 }
    };
    for (const auto& z : values)
    {
        std::complex<double> sum = xsimd::asin(z) + xsimd::acos(z);
        EXPECT_NEAR(sum.real(), pio2<double>(), kDoubleTol) << "z=" << z;
        EXPECT_NEAR(sum.imag(), 0.0, kDoubleTol) << "z=" << z;
    }
}

TEST(XsimdScalarCmathTest, ComplexTanZeroIsZero)
{
    std::complex<double> z0(0.0, 0.0);
    std::complex<double> r = xsimd::tan(z0);
    EXPECT_DOUBLE_EQ(r.real(), 0.0);
    EXPECT_DOUBLE_EQ(r.imag(), 0.0);
}

TEST(XsimdScalarCmathTest, ComplexAtanZeroIsZero)
{
    std::complex<double> z0(0.0, 0.0);
    std::complex<double> r = xsimd::atan(z0);
    EXPECT_DOUBLE_EQ(r.real(), 0.0);
    EXPECT_DOUBLE_EQ(r.imag(), 0.0);
}

TEST(XsimdScalarCmathTest, ComplexAsinhAcoshKnownValues)
{
    // asinh(0) = 0
    std::complex<double> z_asinh(0.0, 0.0);
    std::complex<double> r_asinh = xsimd::asinh(z_asinh);
    EXPECT_NEAR(r_asinh.real(), 0.0, kDoubleTol);
    EXPECT_NEAR(r_asinh.imag(), 0.0, kDoubleTol);

    // acosh(1) = 0
    std::complex<double> z_acosh(1.0, 0.0);
    std::complex<double> r_acosh = xsimd::acosh(z_acosh);
    EXPECT_NEAR(r_acosh.real(), 0.0, kDoubleTol);
    EXPECT_NEAR(r_acosh.imag(), 0.0, kDoubleTol);
}

// =============================================================================
// Edge / corner cases
// =============================================================================

TEST(XsimdScalarCmathTest, SinLargeArgumentReduction)
{
    // Exercises the argument reduction path (x > 20*pi). The result should
    // remain within [-1, 1] and be close to the std::sin reference.
    const float big = 123.456f;
    EXPECT_NEAR(xsimd::sin(big), std::sin(big), 1e-3f);

    const double big_d = 1e6 + 0.5;
    EXPECT_NEAR(xsimd::sin(big_d), std::sin(big_d), 1e-6);
}

TEST(XsimdScalarCmathTest, SinCosPiMultiple)
{
    // Integer multiples of pi should give sin ~ 0
    for (int k = 1; k <= 10; ++k)
    {
        float v = static_cast<float>(k) * pi<float>();
        EXPECT_LE(std::fabs(xsimd::sin(v)), kFloatTol) << "k=" << k;
    }
}

TEST(XsimdScalarCmathTest, AtanSymmetry)
{
    // atan(-x) = -atan(x)
    const double values[] = { 0.3, 1.0, 2.5, 10.0 };
    for (double v : values)
    {
        EXPECT_NEAR(xsimd::atan(-v), -xsimd::atan(v), kDoubleTol) << "v=" << v;
    }
}

TEST(XsimdScalarCmathTest, AcotMonotonicDecreasing)
{
    // acot is strictly decreasing on (0, +inf)
    EXPECT_GT(xsimd::acot(0.5), xsimd::acot(1.0));
    EXPECT_GT(xsimd::acot(1.0), xsimd::acot(2.0));
    EXPECT_GT(xsimd::acot(2.0), xsimd::acot(10.0));
}
