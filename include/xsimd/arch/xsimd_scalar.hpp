/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_SCALAR_HPP
#define XSIMD_SCALAR_HPP

#include "../config/xsimd_macros.hpp"
#include "./xsimd_scalar_math.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include <xtl/xcomplex.hpp>
#endif

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#endif

namespace xsimd
{


    // Floating-point overloads of the core math functions are routed to
    // the libm-free scalar kernels in xsimd_scalar_math.hpp (namespace
    // xsimd::scalar), so that user code calling xsimd::sin etc. does not
    // pull in libm symbols. Other argument types (long double, complex,
    // integral promoted ones) fall back to the standard library.

    template <typename T>
    XSIMD_INLINE T cos(T const& v) noexcept
    {
        return scalar::cos(v);
    }

    template <typename T>
    XSIMD_INLINE T cosh(T const& v) noexcept
    {
        return scalar::cosh(v);
    }

    template <typename T>
    XSIMD_INLINE T exp(T const& v) noexcept
    {
        return scalar::exp(v);
    }

    template <typename T>
    XSIMD_INLINE T log(T const& v) noexcept
    {
        return scalar::log(v);
    }

    template <typename T>
    XSIMD_INLINE T sin(T const& v) noexcept
    {
        return scalar::sin(v);
    }

    template <typename T>
    XSIMD_INLINE T sinh(T const& v) noexcept
    {
        return scalar::sinh(v);
    }

    template <typename T>
    XSIMD_INLINE T tan(T const& v) noexcept
    {
        return scalar::tan(v);
    }

    template <typename T>
    XSIMD_INLINE T tanh(T const& v) noexcept
    {
        return scalar::tanh(v);
    }


    // Real float/double overloads of the remaining core math functions,
    // routed to the libm-free scalar kernels.

    template <typename T>
    XSIMD_INLINE T asin(T const& v) noexcept
    {
        return scalar::asin(v);
    }

    template <typename T>
    XSIMD_INLINE T acos(T const& v) noexcept
    {
        return scalar::acos(v);
    }

    template <typename T>
    XSIMD_INLINE T acosh(T const& v) noexcept
    {
        return scalar::acosh(v);
    }

    template <typename T>
    XSIMD_INLINE T asinh(T const& v) noexcept
    {
        return scalar::asinh(v);
    }

    template <typename T>
    XSIMD_INLINE T atan(T const& v) noexcept
    {
        return scalar::atan(v);
    }

    template <typename T>
    XSIMD_INLINE T atanh(T const& v) noexcept
    {
        return scalar::atanh(v);
    }

    template <typename T>
    XSIMD_INLINE T exp2(T const& v) noexcept
    {
        return scalar::exp2(v);
    }

    template <typename T>
    XSIMD_INLINE T expm1(T const& v) noexcept
    {
        return scalar::expm1(v);
    }

    template <typename T>
    XSIMD_INLINE T log1p(T const& v) noexcept
    {
        return scalar::log1p(v);
    }

    template <typename T>
    XSIMD_INLINE T log2(T const& v) noexcept
    {
        return scalar::log2(v);
    }

    template <typename T>
    XSIMD_INLINE T log10(T const& v) noexcept
    {
        return scalar::log10(v);
    }

    template <typename T0, typename T1>
    XSIMD_INLINE std::common_type_t<T0, T1> atan2(T0 const& y, T1 const& x) noexcept
    {
        return scalar::atan2(y, x);
    }

    template <typename T>
    XSIMD_INLINE T ceil(T const& v) noexcept
    {
        return scalar::detail::ceil_scalar(v);
    }

    template <typename T>
    XSIMD_INLINE T nextafter(T const& from, T const& to) noexcept
    {
        return scalar::detail::nextafter_scalar(from, to);
    }

    template <typename T>
    XSIMD_INLINE T sqrt(T const& v) noexcept
    {
        return scalar::sqrt(v);
    }

    // Routed to the libm-free scalar kernels for float/double; complex and
    // other value types are handled by the overloads below or by the
    // scalar kernels directly.
    template <typename T>
    XSIMD_INLINE T fabs(T const& v) noexcept
    {
        return scalar::detail::abs_scalar(v);
    }

    // Floating-point classification, implemented on the bit patterns so
    // nothing is routed through libm.
    template <typename T>
    XSIMD_INLINE bool isnan(T const& v) noexcept
    {
        return v != v;
    }

    template <typename T>
    XSIMD_INLINE bool isinf(T const& v) noexcept
    {
        return v == scalar::detail::inf_scalar(T()) || v == -scalar::detail::inf_scalar(T());
    }

    template <typename T>
    XSIMD_INLINE bool isfinite(T const& v) noexcept
    {
        return !isnan(v) && !isinf(v);
    }

    template <typename T>
    XSIMD_INLINE T copysign(T const& x, T const& y) noexcept
    {
        return scalar::copysign(x, y);
    }

    template <typename T>
    XSIMD_INLINE T ldexp(T const& v, int e) noexcept
    {
        return scalar::detail::ldexp_scalar(v, static_cast<as_integer_t<T>>(e));
    }

    template <typename T>
    XSIMD_INLINE T modf(T const& v, T* iptr) noexcept
    {
        // mirror of std::modf: integer part with the sign of x
        T i = scalar::detail::trunc_scalar(v);
        *iptr = i;
        return v - i;
    }

    template <typename T>
    XSIMD_INLINE T hypot(T const& x, T const& y) noexcept
    {
        return scalar::detail::hypot_scalar(x, y);
    }

    template <typename T>
    XSIMD_INLINE T cbrt(T const& v) noexcept
    {
        return scalar::cbrt(v);
    }

    namespace detail
    {
        // For integral arguments the rounding / comparison helpers degenerate
        // to the identity or plain arithmetic; only floating-point types go
        // through the bit-level scalar kernels.
        template <typename T>
        using is_fp = std::integral_constant<bool, std::is_same<T, float>::value || std::is_same<T, double>::value>;
    }

    // One macro per shape; the FP branch evaluates a kernel expression,
    // the integral branch is plain arithmetic (rounding of an integer is
    // the identity), mirroring what std:: provides via overloads.
#define XSIMD_SCALAR_UNARY_FP_INTEGRAL(name, fp_expr, int_expr)               \
    template <typename T>                                                     \
    XSIMD_INLINE std::enable_if_t<detail::is_fp<T>::value, T>                  \
    name(T const& v) noexcept                                                 \
    {                                                                         \
        return fp_expr;                                                       \
    }                                                                         \
    template <typename T>                                                     \
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>               \
    name(T const& v) noexcept                                                 \
    {                                                                         \
        return int_expr;                                                      \
    }

#define XSIMD_SCALAR_BINARY_FP_INTEGRAL(name, fp_expr, int_expr)              \
    template <typename T>                                                     \
    XSIMD_INLINE std::enable_if_t<detail::is_fp<T>::value, T>                  \
    name(T const& x, T const& y) noexcept                                     \
    {                                                                         \
        return fp_expr;                                                       \
    }                                                                         \
    template <typename T>                                                     \
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>               \
    name(T const& x, T const& y) noexcept                                     \
    {                                                                         \
        return int_expr;                                                      \
    }

    XSIMD_SCALAR_UNARY_FP_INTEGRAL(trunc, scalar::detail::trunc_scalar(v), v)
    XSIMD_SCALAR_UNARY_FP_INTEGRAL(floor, scalar::detail::floor_scalar(v), v)
    XSIMD_SCALAR_UNARY_FP_INTEGRAL(round, scalar::round(v), v)
    XSIMD_SCALAR_UNARY_FP_INTEGRAL(rint, scalar::detail::nearbyint_scalar(v), v)
    XSIMD_SCALAR_UNARY_FP_INTEGRAL(nearbyint, scalar::detail::nearbyint_scalar(v), v)
    XSIMD_SCALAR_BINARY_FP_INTEGRAL(fdim, scalar::fdim(x, y), x > y ? T(x - y) : T(0))
    XSIMD_SCALAR_BINARY_FP_INTEGRAL(fmax, scalar::fmax(x, y), x > y ? x : y)
    XSIMD_SCALAR_BINARY_FP_INTEGRAL(fmin, scalar::fmin(x, y), x < y ? x : y)
    XSIMD_SCALAR_BINARY_FP_INTEGRAL(fmod, scalar::fmod(x, y), x % y)
    XSIMD_SCALAR_BINARY_FP_INTEGRAL(remainder, scalar::remainder(x, y), x % y)

#undef XSIMD_SCALAR_UNARY_FP_INTEGRAL
#undef XSIMD_SCALAR_BINARY_FP_INTEGRAL

    // abs: floating-point magnitude via the bit-level kernel, complex
    // modulus via hypot (mirror of kernel::abs). Integer overloads are
    // defined further below in this header.
    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    abs(T const& v) noexcept
    {
        return scalar::detail::abs_scalar(v);
    }

    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    abs(const std::complex<T>& v) noexcept
    {
        return scalar::detail::hypot_scalar(v.real(), v.imag());
    }

    // arg: complex argument via atan2; real arguments yield 0 or -0
    // (sign of the input) or pi for negatives, mirroring std::arg.
    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    arg(const std::complex<T>& v) noexcept
    {
        return scalar::atan2(v.imag(), v.real());
    }

    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    arg(T const& v) noexcept
    {
        // atan2(+0, x) = 0 for x >= 0, pi for x < 0; atan2(-0, x) keeps -0/pi
        return scalar::atan2(T(0.), v);
    }

    // conj: complex conjugate; real arguments convert to a complex with
    // zero (negated zero) imaginary part, mirroring std::conj.
    template <typename T>
    XSIMD_INLINE std::complex<T> conj(const std::complex<T>& v) noexcept
    {
        return std::complex<T>(v.real(), -v.imag());
    }

    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, std::complex<T>>
    conj(T const& v) noexcept
    {
        return std::complex<T>(v, -T(0.));
    }

    // norm: re^2 + im^2 (mirror of the batch kernel); real arguments
    // yield x^2, mirroring std::norm.
    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    norm(const std::complex<T>& v) noexcept
    {
        return scalar::detail::fma_scalar(v.real(), v.real(), v.imag() * v.imag());
    }

    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    norm(T const& v) noexcept
    {
        return v * v;
    }

    // polar: rho * (cos theta, sin theta).
    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, std::complex<T>>
    polar(T const& rho, T theta = T(0.)) noexcept
    {
        return std::complex<T>(rho * scalar::cos(theta), rho * scalar::sin(theta));
    }

    // proj: mirror of the batch projection onto the Riemann sphere; real
    // arguments convert to complex first, mirroring std::proj.
    template <typename T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, std::complex<T>>
    proj(T const& v) noexcept
    {
        return proj(std::complex<T>(v, T(0.)));
    }

    template <typename T>
    XSIMD_INLINE std::complex<T> proj(const std::complex<T>& v) noexcept
    {
        T inf = scalar::detail::inf_scalar(T());
        bool isinf_re = v.real() == inf || v.real() == -inf;
        bool isinf_im = v.imag() == inf || v.imag() == -inf;
        if (isinf_re || isinf_im)
            return std::complex<T>(inf, v.imag() < T(0.) ? -T(0.) : T(0.));
        return v;
    }

    // erf / erfc / lgamma / tgamma route to the scalar gamma kernels
    // (mirrors of the batch error/gamma functions).
    template <typename T>
    XSIMD_INLINE T erf(T const& v) noexcept
    {
        return scalar::erf(v);
    }

    template <typename T>
    XSIMD_INLINE T erfc(T const& v) noexcept
    {
        return scalar::erfc(v);
    }

    template <typename T>
    XSIMD_INLINE T lgamma(T const& v) noexcept
    {
        return scalar::lgamma(v);
    }

    template <typename T>
    XSIMD_INLINE T tgamma(T const& v) noexcept
    {
        return scalar::tgamma(v);
    }

    template <typename T>
    XSIMD_INLINE constexpr std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, T>
    abs(T v) noexcept
    {
        return v < 0 ? -v : v;
    }

    template <typename T>
    XSIMD_INLINE constexpr std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, T>
    abs(T v) noexcept
    {
        return v;
    }

// Windows needs the same classification helpers; the generic templates
// above (bit-level, libm-free) already cover both floating-point and,
// via promotion, integral arguments, so no platform split is needed.

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> add(T const& x, Tp const& y) noexcept
    {
        return x + y;
    }

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> avg(T const& x, Tp const& y) noexcept
    {
        using common_type = std::common_type_t<T, Tp>;
        if (std::is_floating_point<common_type>::value)
            return (x + y) / 2;
        else if (std::is_unsigned<common_type>::value)
        {
            return (x & y) + ((x ^ y) >> 1);
        }
        else
        {
            // Inspired by
            // https://stackoverflow.com/questions/5697500/take-the-average-of-two-signed-numbers-in-c
            auto t = (x & y) + ((x ^ y) >> 1);
            auto t_u = static_cast<std::make_unsigned_t<common_type>>(t);
            auto avg = t + (static_cast<T>(t_u >> (8 * sizeof(T) - 1)) & (x ^ y));
            return avg;
        }
    }

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> avgr(T const& x, Tp const& y) noexcept
    {
        using common_type = std::common_type_t<T, Tp>;
        if (std::is_floating_point<common_type>::value)
            return avg(x, y);
        else
        {
            return avg(x, y) + ((x ^ y) & 1);
        }
    }

    template <class T>
    XSIMD_INLINE T incr(T const& x) noexcept
    {
        return x + T(1);
    }

    template <class T>
    XSIMD_INLINE T incr_if(T const& x, bool mask) noexcept
    {
        return x + T(mask ? 1 : 0);
    }

    XSIMD_INLINE bool all(bool mask)
    {
        return mask;
    }

    XSIMD_INLINE bool any(bool mask)
    {
        return mask;
    }

    XSIMD_INLINE bool none(bool mask)
    {
        return !mask;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_and(T x, T y) noexcept
    {
        return x & y;
    }

    template <class T_out, class T_in>
    XSIMD_INLINE T_out bitwise_cast(T_in x) noexcept
    {
        static_assert(sizeof(T_in) == sizeof(T_out), "bitwise_cast between types of the same size");
        T_out r;
        std::memcpy((void*)&r, (void*)&x, sizeof(T_in));
        return r;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T>
    bitwise_and(T x, T y) noexcept
    {
        using I = as_unsigned_integer_t<T>;
        I ix, iy, ir;
        std::memcpy((void*)&ix, (void*)&x, sizeof(T));
        std::memcpy((void*)&iy, (void*)&y, sizeof(T));
        ir = ix & iy;
        T r;
        std::memcpy((void*)&r, (void*)&ir, sizeof(T));
        return r;
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value, T0>
    bitwise_lshift(T0 x, T1 shift) noexcept
    {
        return x << shift;
    }

    template <size_t shift, class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_lshift(T x) noexcept
    {
        constexpr auto bits = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;
        static_assert(shift < bits, "Count must be less than the number of bits in T");
        return x << shift;
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value, T0>
    bitwise_rshift(T0 x, T1 shift) noexcept
    {
        return x >> shift;
    }
    template <size_t shift, class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_rshift(T x) noexcept
    {
        constexpr auto bits = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;
        static_assert(shift < bits, "Count must be less than the number of bits in T");
        return x >> shift;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_not(T x) noexcept
    {
        return ~x;
    }

    XSIMD_INLINE bool bitwise_not(bool x) noexcept
    {
        return !x;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T>
    bitwise_not(T x) noexcept
    {
        using I = as_unsigned_integer_t<T>;
        I ix, ir;
        std::memcpy((void*)&ix, (void*)&x, sizeof(T));
        ir = ~ix;
        T r;
        std::memcpy((void*)&r, (void*)&ir, sizeof(T));
        return r;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_scalar<T>::value, T> bitwise_andnot(T x, T y) noexcept
    {
        return bitwise_and(x, bitwise_not(y));
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_or(T x, T y) noexcept
    {
        return x | y;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T>
    bitwise_or(T x, T y) noexcept
    {
        using I = as_unsigned_integer_t<T>;
        I ix, iy, ir;
        std::memcpy((void*)&ix, (void*)&x, sizeof(T));
        std::memcpy((void*)&iy, (void*)&y, sizeof(T));
        ir = ix | iy;
        T r;
        std::memcpy((void*)&r, (void*)&ir, sizeof(T));
        return r;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    bitwise_xor(T x, T y) noexcept
    {
        return x ^ y;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T>
    bitwise_xor(T x, T y) noexcept
    {
        using I = as_unsigned_integer_t<T>;
        I ix, iy, ir;
        std::memcpy((void*)&ix, (void*)&x, sizeof(T));
        std::memcpy((void*)&iy, (void*)&y, sizeof(T));
        ir = ix ^ iy;
        T r;
        std::memcpy((void*)&r, (void*)&ir, sizeof(T));
        return r;
    }

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> div(T const& x, Tp const& y) noexcept
    {
        return x / y;
    }

    template <class T, class Tp>
    XSIMD_INLINE auto mod(T const& x, Tp const& y) noexcept
    {
        return x % y;
    }

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> mul(T const& x, Tp const& y) noexcept
    {
        return x * y;
    }

    template <class T>
    XSIMD_INLINE T neg(T const& x) noexcept
    {
        return -x;
    }

    template <class T>
    XSIMD_INLINE auto pos(T const& x) noexcept
    {
        return +x;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> reciprocal(T const& x) noexcept
    {
        return T(1.) / x;
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value, T0>
    rotl(T0 x, T1 shift) noexcept
    {
        constexpr auto bits = std::numeric_limits<T0>::digits + std::numeric_limits<T0>::is_signed;
        return (x << shift) | (x >> (bits - shift));
    }
    template <size_t count, class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    rotl(T x) noexcept
    {
        constexpr auto bits = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;
        static_assert(count < bits, "Count must be less than the number of bits in T");
        return (x << count) | (x >> (bits - count));
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value, T0>
    rotr(T0 x, T1 shift) noexcept
    {
        constexpr auto bits = std::numeric_limits<T0>::digits + std::numeric_limits<T0>::is_signed;
        return (x >> shift) | (x << (bits - shift));
    }
    template <size_t count, class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T>
    rotr(T x) noexcept
    {
        constexpr auto bits = std::numeric_limits<T>::digits + std::numeric_limits<T>::is_signed;
        static_assert(count < bits, "Count must be less than the number of bits in T");
        return (x >> count) | (x << (bits - count));
    }

    template <class T>
    XSIMD_INLINE bool isnan(std::complex<T> var) noexcept
    {
        return xsimd::isnan(std::real(var)) || xsimd::isnan(std::imag(var));
    }

    template <class T>
    XSIMD_INLINE bool isinf(std::complex<T> var) noexcept
    {
        return xsimd::isinf(std::real(var)) || xsimd::isinf(std::imag(var));
    }

    template <class T>
    XSIMD_INLINE bool isfinite(std::complex<T> var) noexcept
    {
        return xsimd::isfinite(std::real(var)) && xsimd::isfinite(std::imag(var));
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    using xtl::abs;
    using xtl::acos;
    using xtl::acosh;
    using xtl::asin;
    using xtl::asinh;
    using xtl::atan;
    using xtl::atanh;
    using xtl::cos;
    using xtl::cosh;
    using xtl::exp;
    using xtl::log;
    using xtl::log10;
    using xtl::norm;
    using xtl::pow;
    using xtl::proj;
    using xtl::sin;
    using xtl::sinh;
    using xtl::sqrt;
    using xtl::tan;
    using xtl::tanh;
#endif

    template <typename T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE T clip(const T& val, const T& low, const T& hi) noexcept
    {
        assert(low <= hi && "ordered clipping bounds");
        return low > val ? low : (hi < val ? hi : val);
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool is_flint(const T& x) noexcept
    {
#ifdef __FAST_MATH__
        return (x - scalar::detail::trunc_scalar(x)) == T(0);
#else
        return (x != x) ? false : (x - scalar::detail::trunc_scalar(x)) == T(0);
#endif
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool is_even(const T& x) noexcept
    {
        return is_flint(x * T(0.5));
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool is_odd(const T& x) noexcept
    {
        return is_even(x - 1.);
    }

    template <class T, class I = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>>
    XSIMD_INLINE I nearbyint_as_int(T var) noexcept
    {
        return static_cast<I>(scalar::detail::nearbyint_scalar(var));
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool eq(const T& x0, const T& x1) noexcept
    {
        return x0 == x1;
    }

    template <class T>
    XSIMD_INLINE bool eq(const std::complex<T>& x0, const std::complex<T>& x1) noexcept
    {
        return x0 == x1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool ge(const T& x0, const T& x1) noexcept
    {
        return x0 >= x1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool gt(const T& x0, const T& x1) noexcept
    {
        return x0 > x1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool le(const T& x0, const T& x1) noexcept
    {
        return x0 <= x1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool lt(const T& x0, const T& x1) noexcept
    {
        return x0 < x1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE bool neq(const T& x0, const T& x1) noexcept
    {
        return x0 != x1;
    }

    template <class T>
    XSIMD_INLINE bool neq(const std::complex<T>& x0, const std::complex<T>& x1) noexcept
    {
        return !(x0 == x1);
    }

    // The generic exp2-based evaluation is off by 1 ulp against the
    // reference std::pow(10, x) on some inputs, and the platform
    // exp10/exp10f expand to inline builtins (no libm symbol), so route
    // through them; platforms without any flavor fall back to the kernel.
#if defined(__GLIBC__)
    XSIMD_INLINE float exp10(const float& x) noexcept
    {
        return ::exp10f(x);
    }
    XSIMD_INLINE double exp10(const double& x) noexcept
    {
        return ::exp10(x);
    }
#else
    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, T>
    exp10(const T& x) noexcept
    {
        T l2t = xsimd::bit_cast<T>(as_unsigned_integer_t<T>(sizeof(T) == 4 ? 0x40549a78u : 0x400a934f0979a371ull));
        return scalar::exp2(x * l2t);
    }
#endif

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE auto rsqrt(const T& x) noexcept
    {
        return static_cast<T>(1) / scalar::detail::sqrt_scalar(x);
    }

    // Complex expm1 / log1p / log2 / log10 route to the scalar complex
    // kernels (xsimd_scalar_math.hpp) via overload resolution; xtl
    // complex keeps its own wrappers over the same expressions.
    template <class T>
    XSIMD_INLINE std::complex<T> expm1(const std::complex<T>& val) noexcept
    {
        return scalar::expm1(val);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> expm1(const xtl::xcomplex<T, T, i3ec>& val) noexcept
    {
        using C = xtl::xcomplex<T, T, i3ec>;
        T isin = xsimd::sin(val.imag());
        T rem1 = xsimd::expm1(val.real());
        T re = rem1 + T(1.);
        T si = xsimd::sin(val.imag() * T(0.5));
        return C(rem1 - T(2.) * re * si * si, re * isin);
    }
#endif

    template <class T>
    XSIMD_INLINE std::complex<T> log1p(const std::complex<T>& val) noexcept
    {
        return scalar::log1p(val);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> log1p(const xtl::xcomplex<T, T, i3ec>& val) noexcept
    {
        return scalar::log1p(std::complex<T>(val.real(), val.imag()));
    }
#endif

    template <class T>
    XSIMD_INLINE std::complex<T> log2(const std::complex<T>& val) noexcept
    {
        return scalar::log2(val);
    }

    template <class T>
    XSIMD_INLINE std::complex<T> log10(const std::complex<T>& val) noexcept
    {
        return scalar::log10(val);
    }

    template <typename T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE T sadd(const T& lhs, const T& rhs) noexcept
    {
        if (std::numeric_limits<T>::is_signed)
        {
            if ((lhs > 0) && (rhs > std::numeric_limits<T>::max() - lhs))
            {
                return std::numeric_limits<T>::max();
            }
            else if ((lhs < 0) && (rhs < std::numeric_limits<T>::lowest() - lhs))
            {
                return std::numeric_limits<T>::lowest();
            }
            else
            {
                return lhs + rhs;
            }
        }
        else
        {
            if (rhs > std::numeric_limits<T>::max() - lhs)
            {
                return std::numeric_limits<T>::max();
            }
            else
            {
                return lhs + rhs;
            }
        }
    }

    template <typename T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE T ssub(const T& lhs, const T& rhs) noexcept
    {
        if (std::numeric_limits<T>::is_signed)
        {
            return sadd(lhs, (T)-rhs);
        }
        else
        {
            if (lhs < rhs)
            {
                return std::numeric_limits<T>::lowest();
            }
            else
            {
                return lhs - rhs;
            }
        }
    }

    namespace detail
    {
        template <class T0, class T1>
        XSIMD_INLINE std::enable_if_t<std::is_integral<T1>::value, T0>
        ipow(const T0& x, const T1& n) noexcept
        {
            static_assert(std::is_integral<T1>::value, "second argument must be an integer");
            T0 a = x;
            T1 b = n;
            bool const recip = b < 0;
            T0 r(static_cast<T0>(1));
            while (1)
            {
                if (b & 1)
                {
                    r *= a;
                }
                b /= 2;
                if (b == 0)
                {
                    break;
                }
                a *= a;
            }
            return recip ? static_cast<T0>(1) / r : r;
        }
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T1>::value, T0>
    pow(const T0& x, const T1& n) noexcept
    {
        return detail::ipow(x, n);
    }

    template <class T0, class T1, class = std::enable_if_t<std::is_scalar<T0>::value && std::is_floating_point<T1>::value>>
    XSIMD_INLINE auto
    pow(const T0& t0, const T1& t1) noexcept
    {
        return scalar::pow(static_cast<T1>(t0), t1);
    }

    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T1>::value, std::complex<T0>>
    pow(const std::complex<T0>& t0, const T1& t1) noexcept
    {
        return detail::ipow(t0, t1);
    }

    // Complex pow routes to the scalar complex kernel; mixed real/complex
    // argument combinations convert first.
    template <class T0, class T1>
    XSIMD_INLINE std::enable_if_t<!std::is_integral<T1>::value, std::complex<T0>>
    pow(const std::complex<T0>& t0, const T1& t1) noexcept
    {
        return scalar::pow(t0, std::complex<T1>(t1, T1(0.)));
    }

    template <class T>
    XSIMD_INLINE std::complex<T>
    pow(const std::complex<T>& t0, const std::complex<T>& t1) noexcept
    {
        return scalar::pow(t0, t1);
    }

    template <class T0, class T1, class = std::enable_if_t<std::is_scalar<T0>::value>>
    XSIMD_INLINE auto
    pow(const T0& t0, const std::complex<T1>& t1) noexcept
    {
        return scalar::pow(std::complex<T1>(static_cast<T1>(t0), T1(0.)), t1);
    }

    // sign bit: comparison for integral / long double, bit pattern for
    // float/double (no libm, and distinguishes -0.0 like std::signbit)
    template <class T, class = std::enable_if_t<std::is_scalar<T>::value && !std::is_same<T, float>::value && !std::is_same<T, double>::value>>
    XSIMD_INLINE T bitofsign(T const& x) noexcept
    {
        return T(x < T(0));
    }

    XSIMD_INLINE float bitofsign(float const& x) noexcept
    {
        using I = as_unsigned_integer_t<float>;
        I b;
        std::memcpy((void*)&b, (void*)&x, sizeof(float));
        return float((b >> 31) & 1);
    }

    XSIMD_INLINE double bitofsign(double const& x) noexcept
    {
        using I = as_unsigned_integer_t<double>;
        I b;
        std::memcpy((void*)&b, (void*)&x, sizeof(double));
        return double((b >> 63) & 1);
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE auto signbit(T const& v) noexcept
    {
        return bitofsign(v);
    }

    XSIMD_INLINE double sign(bool const& v) noexcept
    {
        return v;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE T sign(const T& v) noexcept
    {
        return v < T(0) ? T(-1.) : v == T(0) ? T(0.)
                                             : T(1.);
    }

    namespace detail
    {
        template <class C>
        XSIMD_INLINE C sign_complex_scalar_impl(const C& v) noexcept
        {
            using value_type = typename C::value_type;
            if (v.real())
            {
                return C(sign(v.real()), value_type(0));
            }
            else
            {
                return C(sign(v.imag()), value_type(0));
            }
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> sign(const std::complex<T>& v) noexcept
    {
        return detail::sign_complex_scalar_impl(v);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> sign(const xtl::xcomplex<T, T, i3ec>& v) noexcept
    {
        return detail::sign_complex_scalar_impl(v);
    }
#endif

    XSIMD_INLINE double signnz(bool const&) noexcept
    {
        return 1;
    }

    template <class T, class = std::enable_if_t<std::is_scalar<T>::value>>
    XSIMD_INLINE T signnz(const T& v) noexcept
    {
        return v < T(0) ? T(-1.) : T(1.);
    }

    template <class T, class Tp>
    XSIMD_INLINE std::common_type_t<T, Tp> sub(T const& x, Tp const& y) noexcept
    {
        return x - y;
    }

    template <class T>
    XSIMD_INLINE T decr(T const& x) noexcept
    {
        return x - T(1);
    }

    template <class T>
    XSIMD_INLINE T decr_if(T const& x, bool mask) noexcept
    {
        return x - T(mask ? 1 : 0);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> log2(const xtl::xcomplex<T, T, i3ec>& val) noexcept
    {
        return log(val) / log(T(2));
    }
#endif

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> log1p(const xtl::xcomplex<T, T, i3ec>& val) noexcept
    {
        return detail::log1p_complex_scalar_impl(val);
    }
#endif

    template <class T0, class T1, class = std::enable_if_t<std::is_scalar<T0>::value && std::is_scalar<T1>::value>>
    XSIMD_INLINE auto min(T0 const& self, T1 const& other) noexcept
    {
        return self > other ? other : self;
    }

    // numpy defines minimum operator on complex using lexical comparison
    template <class T0, class T1>
    XSIMD_INLINE std::complex<std::common_type_t<T0, T1>>
    min(std::complex<T0> const& self, std::complex<T1> const& other) noexcept
    {
        return (self.real() < other.real()) ? (self) : (self.real() == other.real() ? (self.imag() < other.imag() ? self : other) : other);
    }

    template <class T0, class T1, class = std::enable_if_t<std::is_scalar<T0>::value && std::is_scalar<T1>::value>>
    XSIMD_INLINE auto max(T0 const& self, T1 const& other) noexcept
    {
        return self < other ? other : self;
    }

    // numpy defines maximum operator on complex using lexical comparison
    template <class T0, class T1>
    XSIMD_INLINE std::complex<std::common_type_t<T0, T1>>
    max(std::complex<T0> const& self, std::complex<T1> const& other) noexcept
    {
        return (self.real() > other.real()) ? (self) : (self.real() == other.real() ? (self.imag() > other.imag() ? self : other) : other);
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T> fma(const T& a, const T& b, const T& c) noexcept
    {
        return a * b + c;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> fma(const T& a, const T& b, const T& c) noexcept
    {
        return scalar::detail::fma_scalar(a, b, c);
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T> fms(const T& a, const T& b, const T& c) noexcept
    {
        return a * b - c;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> fms(const T& a, const T& b, const T& c) noexcept
    {
        return scalar::detail::fms_scalar(a, b, c);
    }

    namespace detail
    {
        template <class C>
        XSIMD_INLINE C fma_complex_scalar_impl(const C& a, const C& b, const C& c) noexcept
        {
            return { fms(a.real(), b.real(), fms(a.imag(), b.imag(), c.real())),
                     fma(a.real(), b.imag(), fma(a.imag(), b.real(), c.imag())) };
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> fma(const std::complex<T>& a, const std::complex<T>& b, const std::complex<T>& c) noexcept
    {
        return detail::fma_complex_scalar_impl(a, b, c);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> fma(const xtl::xcomplex<T, T, i3ec>& a, const xtl::xcomplex<T, T, i3ec>& b, const xtl::xcomplex<T, T, i3ec>& c) noexcept
    {
        return detail::fma_complex_scalar_impl(a, b, c);
    }
#endif

    namespace detail
    {
        template <class C>
        XSIMD_INLINE C fms_complex_scalar_impl(const C& a, const C& b, const C& c) noexcept
        {
            return { fms(a.real(), b.real(), fma(a.imag(), b.imag(), c.real())),
                     fma(a.real(), b.imag(), fms(a.imag(), b.real(), c.imag())) };
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> fms(const std::complex<T>& a, const std::complex<T>& b, const std::complex<T>& c) noexcept
    {
        return detail::fms_complex_scalar_impl(a, b, c);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> fms(const xtl::xcomplex<T, T, i3ec>& a, const xtl::xcomplex<T, T, i3ec>& b, const xtl::xcomplex<T, T, i3ec>& c) noexcept
    {
        return detail::fms_complex_scalar_impl(a, b, c);
    }
#endif

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T> fnma(const T& a, const T& b, const T& c) noexcept
    {
        return -(a * b) + c;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> fnma(const T& a, const T& b, const T& c) noexcept
    {
        return scalar::detail::fnma_scalar(a, b, c);
    }

    namespace detail
    {
        template <class C>
        XSIMD_INLINE C fnma_complex_scalar_impl(const C& a, const C& b, const C& c) noexcept
        {
            return { fms(a.imag(), b.imag(), fms(a.real(), b.real(), c.real())),
                     -fma(a.real(), b.imag(), fms(a.imag(), b.real(), c.imag())) };
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> fnma(const std::complex<T>& a, const std::complex<T>& b, const std::complex<T>& c) noexcept
    {
        return detail::fnma_complex_scalar_impl(a, b, c);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> fnma(const xtl::xcomplex<T, T, i3ec>& a, const xtl::xcomplex<T, T, i3ec>& b, const xtl::xcomplex<T, T, i3ec>& c) noexcept
    {
        return detail::fnma_complex_scalar_impl(a, b, c);
    }
#endif

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value, T> fnms(const T& a, const T& b, const T& c) noexcept
    {
        return -(a * b) - c;
    }

    template <class T>
    XSIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> fnms(const T& a, const T& b, const T& c) noexcept
    {
        return -scalar::detail::fma_scalar(a, b, c);
    }

    namespace detail
    {
        template <class C>
        XSIMD_INLINE C fnms_complex_scalar_impl(const C& a, const C& b, const C& c) noexcept
        {
            return { fms(a.imag(), b.imag(), fma(a.real(), b.real(), c.real())),
                     -fma(a.real(), b.imag(), fma(a.imag(), b.real(), c.imag())) };
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> fnms(const std::complex<T>& a, const std::complex<T>& b, const std::complex<T>& c) noexcept
    {
        return detail::fnms_complex_scalar_impl(a, b, c);
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T, bool i3ec>
    XSIMD_INLINE xtl::xcomplex<T, T, i3ec> fnms(const xtl::xcomplex<T, T, i3ec>& a, const xtl::xcomplex<T, T, i3ec>& b, const xtl::xcomplex<T, T, i3ec>& c) noexcept
    {
        return detail::fnms_complex_scalar_impl(a, b, c);
    }
#endif

    namespace detail
    {
#define XSIMD_HASSINCOS_TRAIT(func)                                                                                                           \
    template <class S>                                                                                                                        \
    struct has##func                                                                                                                          \
    {                                                                                                                                         \
        template <class T>                                                                                                                    \
        static XSIMD_INLINE auto get(T* ptr) -> decltype(func(std::declval<T>(), std::declval<T*>(), std::declval<T*>()), std::true_type {}); \
        static XSIMD_INLINE std::false_type get(...);                                                                                         \
        static constexpr bool value = decltype(get((S*)nullptr))::value;                                                                      \
    }

#define XSIMD_HASSINCOS(func, T) has##func<T>::value

        XSIMD_HASSINCOS_TRAIT(sincos);
        XSIMD_HASSINCOS_TRAIT(sincosf);
        XSIMD_HASSINCOS_TRAIT(__sincos);
        XSIMD_HASSINCOS_TRAIT(__sincosf);

        struct common_sincosf
        {
            template <class T>
            XSIMD_INLINE std::enable_if_t<XSIMD_HASSINCOS(sincosf, T)>
            operator()(float val, T& s, T& c)
            {
                sincosf(val, &s, &c);
            }

            template <class T>
            XSIMD_INLINE std::enable_if_t<!XSIMD_HASSINCOS(sincosf, T) && XSIMD_HASSINCOS(__sincosf, T)>
            operator()(float val, T& s, T& c)
            {
                __sincosf(val, &s, &c);
            }

            template <class T>
            XSIMD_INLINE std::enable_if_t<!XSIMD_HASSINCOS(sincosf, T) && !XSIMD_HASSINCOS(__sincosf, T)>
            operator()(float val, T& s, T& c)
            {
                s = xsimd::sin(val);
                c = xsimd::cos(val);
            }
        };

        struct common_sincos
        {
            template <class T>
            XSIMD_INLINE std::enable_if_t<XSIMD_HASSINCOS(sincos, T)>
            operator()(double val, T& s, T& c)
            {
                sincos(val, &s, &c);
            }

            template <class T>
            XSIMD_INLINE std::enable_if_t<!XSIMD_HASSINCOS(sincos, T) && XSIMD_HASSINCOS(__sincos, T)>
            operator()(double val, T& s, T& c)
            {
                __sincos(val, &s, &c);
            }

            template <class T>
            XSIMD_INLINE std::enable_if_t<!XSIMD_HASSINCOS(sincos, T) && !XSIMD_HASSINCOS(__sincos, T)>
            operator()(double val, T& s, T& c)
            {
                s = xsimd::sin(val);
                c = xsimd::cos(val);
            }
        };

#undef XSIMD_HASSINCOS_TRAIT
#undef XSIMD_HASSINCOS
    }

    XSIMD_INLINE std::pair<float, float> sincos(float val) noexcept
    {
        float s, c;
        detail::common_sincosf {}(val, s, c);
        return std::make_pair(s, c);
    }

    XSIMD_INLINE std::pair<double, double> sincos(double val) noexcept
    {
        double s, c;
        detail::common_sincos {}(val, s, c);
        return std::make_pair(s, c);
    }

    template <class T>
    XSIMD_INLINE std::pair<std::complex<T>, std::complex<T>>
    sincos(const std::complex<T>& val) noexcept
    {
        return std::make_pair(xsimd::sin(val), xsimd::cos(val));
    }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    template <class T>
    XSIMD_INLINE std::pair<xtl::xcomplex<T>, xtl::xcomplex<T>> sincos(const xtl::xcomplex<T>& val) noexcept
    {
        return std::make_pair(sin(val), cos(val));
    }
#endif

    namespace detail
    {
        // Bit-level frexp mirror (no libm): normalize the mantissa into
        // [0.5, 1) and return the binary exponent.
        template <class T>
        XSIMD_INLINE T frexp_scalar_impl(T val, int& exp) noexcept
        {
            using I = as_integer_t<T>;
            const int mant_bits = sizeof(T) == 4 ? 23 : 52;
            const int sign_bit = sizeof(T) == 4 ? 31 : 63;
            const I bias = sizeof(T) == 4 ? I(127) : I(1023);
            if (val == T(0.) || val != val || val == scalar::detail::inf_scalar(T()) || val == -scalar::detail::inf_scalar(T()))
            {
                exp = 0;
                return val;
            }
            I i = bit_cast<I>(val);
            I sign = i & (I(1) << sign_bit);
            I e = (i >> mant_bits) & (sizeof(T) == 4 ? I(0xff) : I(0x7ff));
            if (e == 0) // subnormal: scale up, decompose, shift back
            {
                const int scale_bits = sizeof(T) == 4 ? 23 : 52;
                T scaled = val * (sizeof(T) == 4 ? T(8388608.) : T(4503599627370496.));
                I si = bit_cast<I>(scaled);
                I se = (si >> mant_bits) & (sizeof(T) == 4 ? I(0xff) : I(0x7ff));
                exp = int(se) - int(bias) + 1 - scale_bits;
                I m = sign | ((si & ((I(1) << mant_bits) - 1)) | ((bias - 1) << mant_bits));
                return bit_cast<T>(m);
            }
            exp = int(e) - int(bias) + 1;
            I m = sign | ((i & ((I(1) << mant_bits) - 1)) | ((bias - 1) << mant_bits));
            return bit_cast<T>(m);
        }
    }

    template <class T, class = std::enable_if_t<std::is_floating_point<T>::value>>
    XSIMD_INLINE T frexp(T const& val, int& exp) noexcept
    {
        return detail::frexp_scalar_impl(val, exp);
    }

    template <class T>
    XSIMD_INLINE T select(bool cond, T const& true_br, T const& false_br) noexcept
    {
        return cond ? true_br : false_br;
    }

    template <class T>
    XSIMD_INLINE constexpr bool batch_bool_cast(bool b) noexcept
    {
        return b;
    }

    template <class T_out, class T_in>
    XSIMD_INLINE constexpr T_out batch_cast(T_in const& val) noexcept
    {
        static_assert(!std::is_same<T_out, bool>::value, "cannot convert to bool, use !x or x != 0");
        return static_cast<T_out>(val);
    }
}

#endif
