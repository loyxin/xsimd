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

#ifndef XSIMD_SCALAR_MATH_HPP
#define XSIMD_SCALAR_MATH_HPP

// Scalar implementations of sin / cos / tan / cot / sinh / cosh / tanh /
// coth / exp / log for float and double.
//
// Design notes
// ------------
//  * These are *standalone scalar algorithms*, not a wrapper that routes
//    through the batch kernels: each function is the element-wise mirror
//    of the corresponding common batch kernel, using the same polynomial
//    coefficients and the same operation order. FMA call sites use
//    own scalar fma helper (compiler builtin on FMA targets, x*y+z
//    otherwise, mirroring the common kernel fallback — identical bits),
//    so results are bit-identical to the batch kernels.
//  * No libm symbol is pulled in: rounding / scaling helpers (nearbyint,
//    floor, ldexp) are implemented with bit manipulation, mirroring the
//    batch implementations; the large-argument trigonometric reduction
//    calls the scalar fdlibm __ieee754_rem_pio2 which is also libm-free.

#include "../config/xsimd_macros.hpp"
#include "../math/xsimd_rem_pio2.hpp"
#include "../types/xsimd_utils.hpp"
#include "xsimd_constants.hpp"

#include <cstdint>

// XSIMD_SCALAR_FAST_MATH: performance knob for the scalar math entry
// points. When defined (or when the compiler is in fast-math mode), the
// scalar kernels skip the special-case handling of NaN / infinity inputs
// (trigo reduction guards, exp overflow saturation, log domain checks).
// This trades IEEE edge-case correctness for speed; results on finite
// inputs are unchanged. Tests comparing against the batch kernels must
// be skipped when this macro is set, because the batch kernels keep
// their own (compiler-driven) fast-math behavior.
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_FAST_MATH)
#define XSIMD_SCALAR_SKIP_SPECIALS
#endif

namespace xsimd
{
    namespace scalar
    {
        namespace detail
        {
            // --------------
            // scalar helpers
            // --------------

            // Empty inline-asm no-op preventing reassociation / contraction
            // of the surrounding expression, mirroring
            // kernel::detail::reassociation_barrier for scalar code paths.
            template <class T>
            XSIMD_INLINE void scalar_reassociation_barrier(T& x) noexcept
            {
#if defined(__GNUC__)
                __asm__ volatile("" : "+x"(x));
#else
                (void)x;
#endif
            }

            template <class T>
            XSIMD_INLINE T sign_bit_of(T x) noexcept
            {
                using bits = as_unsigned_integer_t<T>;
                bits b = bit_cast<bits>(x);
                b &= bits(1) << (8 * sizeof(T) - 1);
                return bit_cast<T>(b);
            }

            template <class T>
            XSIMD_INLINE T flip_sign(T x, T mask) noexcept
            {
                return bit_cast<T>(bit_cast<as_unsigned_integer_t<T>>(x)
                                   ^ bit_cast<as_unsigned_integer_t<T>>(mask));
            }

            // Bit-level abs, mirrors kernel::abs for float/double
            // (preserves the payload of -0.0 -> +0.0 exactly).
            template <class T>
            XSIMD_INLINE T abs_scalar(T x) noexcept
            {
                return flip_sign(x, sign_bit_of(x));
            }

            // NaN with every bit set, mirroring constants::nan<batch>
            // (0xffffffff / 0xffffffffffffffff) so that sign of NaN
            // results matches the batch kernels bit-for-bit.
            template <class T>
            XSIMD_INLINE T nan_like_batch() noexcept
            {
                return bit_cast<T>(~as_unsigned_integer_t<T>(0));
            }

            // Quiet NaN (0x7fc00000 / 0x7ff8000000000000): exponent all
            // ones + leading mantissa bit, identical to
            // numeric_limits<T>::quiet_NaN() which the batch fallback
            // uses for infinite trigonometric arguments.
            template <class T>
            XSIMD_INLINE T quiet_nan_scalar(T = T()) noexcept
            {
                return bit_cast<T>(as_unsigned_integer_t<T>(sizeof(T) == 4 ? 0x7fc00000u : 0x7ff8000000000000ull));
            }

            // +infinity (0x7f800000 / 0x7ff0000000000000): exponent all
            // ones, mantissa zero.
            template <class T>
            XSIMD_INLINE T inf_scalar(T = T()) noexcept
            {
                return bit_cast<T>(as_unsigned_integer_t<T>(sizeof(T) == 4 ? 0x7f800000u : 0x7ff0000000000000ull));
            }

            // Mirror of kernel::detail::nearbyintf (bit-level, libm-free).
            template <class T>
            XSIMD_INLINE T nearbyint_scalar(T v) noexcept
            {
                T s = sign_bit_of(v);
                T a = flip_sign(v, s); // |v|
                // t2n = 2^nmb (8388608 for float, 2^52 for double), built by
                // bit pattern (libm-free); mirrors constants::twotonmb.
                as_unsigned_integer_t<T> bits_t2n = as_unsigned_integer_t<T>(xsimd::constants::maxexponent<T>() + xsimd::constants::nmb<T>())
                                                    << xsimd::constants::nmb<T>();
                T t2n = bit_cast<T>(bits_t2n);
                if (a < t2n)
                {
                    T d0 = a + t2n;
                    return flip_sign(d0 - t2n, s);
                }
                return v;
            }

            // Mirror of kernel::trunc (libm-free). maxflint = 2^nmb.
            template <class T>
            XSIMD_INLINE T trunc_scalar(T x) noexcept
            {
                T a = x < T(0) ? -x : x;
                // maxflint = 2^nmb built by bit pattern (libm-free)
                as_unsigned_integer_t<T> bits_max = as_unsigned_integer_t<T>(xsimd::constants::maxexponent<T>() + xsimd::constants::nmb<T>())
                                                    << xsimd::constants::nmb<T>();
                T maxflint = bit_cast<T>(bits_max);
                if (a >= maxflint)
                    return x;
                // conversion truncates toward zero
                return static_cast<T>(static_cast<as_integer_t<T>>(x));
            }

            // Mirror of kernel::floor via trunc (bit-level, libm-free).
            template <class T>
            XSIMD_INLINE T floor_scalar(T x) noexcept
            {
                T t = trunc_scalar(x);
                return t > x ? t - T(1) : t;
            }
            // Mirror of kernel::ceil via trunc (bit-level, libm-free).
            template <class T>
            XSIMD_INLINE T ceil_scalar(T x) noexcept
            {
                T t = trunc_scalar(x);
                return t < x ? t + T(1) : t;
            }

            // Mirror of kernel::nextafter (bit-level, libm-free): step the
            // IEEE754 bit pattern of `from` toward `to`.
            template <class T>
            XSIMD_INLINE T nextafter_scalar(T from, T to) noexcept
            {
                using i_type = as_integer_t<T>;
                if (from == to)
                    return from;
                bool up = to > from;
                i_type ib = bit_cast<i_type>(from);
                // signbit(from): negative iff the integer pattern is negative
                bool negative = ib < i_type(0);
                // step direction on the integer representation: toward
                // larger magnitude increments/decrements the magnitude bits,
                // which is ++ib for positive and --ib for negative patterns,
                // but the sign flip maps ++ <-> --; stepping "up" (toward
                // +inf) means ++ib for non-negative patterns and --ib for
                // negative ones.
                i_type stepped = up ? (negative ? ib - i_type(1) : ib + i_type(1))
                                    : (negative ? ib + i_type(1) : ib - i_type(1));
                T r = bit_cast<T>(stepped);
#ifndef __FAST_MATH__
                // mirror the kernel's infinity guards
                if (from == detail::inf_scalar(T()) || from == -detail::inf_scalar(T()))
                    return from;
#endif
                return r;
            }

            // Mirror of kernel::ldexp (bit-level, libm-free).
            template <class T>
            XSIMD_INLINE T ldexp_scalar(T x, as_integer_t<T> e) noexcept
            {
                using i_type = as_integer_t<T>;
                i_type ik = e + xsimd::constants::maxexponent<T>();
                ik <<= xsimd::constants::nmb<T>();
                using u_type = as_unsigned_integer_t<T>;
                return x * xsimd::bit_cast<T>(static_cast<u_type>(ik));
            }

            // Mirror of kernel::detail::horner / horner1, using the scalar
            // fma helper below (no std:: / libm routing).
            // Typed FMA helper: uses the hardware instruction when the
            // type has fast FMA support, otherwise the same x * y + z
            // fallback as the common batch kernel (bit-exact either way).
            template <class T>
            XSIMD_INLINE T fma_scalar(T x, T y, T z) noexcept
            {
                if constexpr (sizeof(T) == 4)
                {
#if defined(__FMA__) || defined(__FP_FAST_FMAF__)
                    return __builtin_fmaf(x, y, z);
#else
                    return x * y + z;
#endif
                }
                else
                {
#if defined(__FMA__) || defined(__FP_FAST_FMA__)
                    return __builtin_fma(x, y, z);
#else
                    return x * y + z;
#endif
                }
            }
            template <class T>
            XSIMD_INLINE T horner_scalar(const T&) noexcept
            {
                return T(0);
            }

            template <class T, uint64_t c0>
            XSIMD_INLINE T coef_scalar() noexcept
            {
                return xsimd::bit_cast<T>(static_cast<as_unsigned_integer_t<T>>(c0));
            }

            template <class T, uint64_t c0>
            XSIMD_INLINE T horner_scalar(const T& x) noexcept
            {
                return coef_scalar<T, c0>();
            }

            template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
            XSIMD_INLINE T horner_scalar(const T& x) noexcept
            {
                return fma_scalar(x, horner_scalar<T, c1, args...>(x), coef_scalar<T, c0>());
            }

            template <class T>
            XSIMD_INLINE T horner1_scalar(const T&) noexcept
            {
                return T(1);
            }

            template <class T, uint64_t c0>
            XSIMD_INLINE T horner1_scalar(const T& x) noexcept
            {
                return x + coef_scalar<T, c0>();
            }

            template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
            XSIMD_INLINE T horner1_scalar(const T& x) noexcept
            {
                return fma_scalar(x, horner1_scalar<T, c1, args...>(x), coef_scalar<T, c0>());
            }

            // Mirror of kernel::detail::quadrant.
            XSIMD_INLINE float quadrant_scalar(float x) noexcept
            {
                // mirrors to_float(quadrant(to_int(x)))
                int32_t i = static_cast<int32_t>(x);
                i &= 3;
                return static_cast<float>(i);
            }

            XSIMD_INLINE double quadrant_scalar(double x) noexcept
            {
                double a = x * 0.25;
                return (a - floor_scalar(a)) * 4.;
            }

            // Mirror of kernel::detail::trigo_reducer (radian tag). The
            // reduction branches are scalar here; the large-argument case
            // calls the libm-free scalar fdlibm __ieee754_rem_pio2.
            template <class T>
            XSIMD_INLINE T trigo_reduce(T x, T& xr) noexcept
            {
                const T pio4 = xsimd::constants::pio4<T>();
                const T pio2_1 = xsimd::constants::pio2_1<T>();
                const T pio2_2 = xsimd::constants::pio2_2<T>();
                const T pio2_3 = xsimd::constants::pio2_3<T>();
                const T pio2_1t = xsimd::constants::pio2_1t<T>();
                const T pio2_2t = xsimd::constants::pio2_2t<T>();
                const T pio2_3t = xsimd::constants::pio2_3t<T>();

                if (x <= pio4)
                {
                    xr = x;
                    return T(0);
                }
                else if (x <= xsimd::constants::pio2<T>())
                {
                    bool test = x > pio4;
                    xr = x - pio2_1;
                    xr -= pio2_2;
                    xr -= pio2_3;
                    xr = test ? xr : x;
                    return test ? T(1) : T(0);
                }
                else if (x <= xsimd::constants::twentypi<T>())
                {
                    T xi = nearbyint_scalar(x * xsimd::constants::twoopi<T>());
                    xr = fma_scalar(-xi, pio2_1, x);
                    xr -= xi * pio2_2;
                    xr -= xi * pio2_3;
                    return quadrant_scalar(xi);
                }
                else if (x <= xsimd::constants::mediumpi<T>())
                {
                    T fn = nearbyint_scalar(x * xsimd::constants::twoopi<T>());
                    T r = x - fn * pio2_1;
                    T w = fn * pio2_1t;
                    T t = r;
                    w = fn * pio2_2;
                    r = t - w;
                    w = fn * pio2_2t - ((t - r) - w);
                    t = r;
                    w = fn * pio2_3;
                    r = t - w;
                    w = fn * pio2_3t - ((t - r) - w);
                    xr = r - w;
                    return quadrant_scalar(fn);
                }
                else
                {
#ifndef XSIMD_SCALAR_SKIP_SPECIALS
                    if (x == detail::inf_scalar(T()))
                    {
                        // mirror the batch scalar-loop fallback, which
                        // assigns quiet_NaN (limits) for infinite args
                        xr = detail::quiet_nan_scalar(T());
                        return T(0);
                    }
#endif
                    double y[2];
                    int32_t n = ::xsimd::detail::__ieee754_rem_pio2(static_cast<double>(x), y);
                    xr = T(y[0]);
                    return T(n & 3);
                }
            }

            // -------------------
            // trigonometric evals
            // -------------------

            XSIMD_INLINE float cos_eval_scalar(float z) noexcept
            {
                float y = horner_scalar<float,
                                        0x3d2aaaa5,
                                        0xbab60619,
                                        0x37ccf5ce>(z);
                return 1.f + fma_scalar(z, -0.5f, y * z * z);
            }

            XSIMD_INLINE float sin_eval_scalar(float z, float x) noexcept
            {
                float y = horner_scalar<float,
                                        0xbe2aaaa2,
                                        0x3c08839d,
                                        0xb94ca1f9>(z);
                return fma_scalar(y * z, x, x);
            }

            XSIMD_INLINE float base_tancot_eval_scalar(float z) noexcept
            {
                float zz = z * z;
                float y = horner_scalar<float,
                                        0x3eaaaa6f,
                                        0x3e0896dd,
                                        0x3d5ac5c9,
                                        0x3cc821b5,
                                        0x3b4c779c,
                                        0x3c19c53b>(zz);
                return fma_scalar(y, zz * z, z);
            }

            XSIMD_INLINE double cos_eval_scalar(double z) noexcept
            {
                double y = horner_scalar<double,
                                         0x3fe0000000000000ull,
                                         0xbfa5555555555551ull,
                                         0x3f56c16c16c15d47ull,
                                         0xbefa01a019ddbcd9ull,
                                         0x3e927e4f8e06d9a5ull,
                                         0xbe21eea7c1e514d4ull,
                                         0x3da8ff831ad9b219ull>(z);
                return 1. - y * z;
            }

            XSIMD_INLINE double sin_eval_scalar(double z, double x) noexcept
            {
                double y = horner_scalar<double,
                                         0xbfc5555555555548ull,
                                         0x3f8111111110f7d0ull,
                                         0xbf2a01a019bfdf03ull,
                                         0x3ec71de3567d4896ull,
                                         0xbe5ae5e5a9291691ull,
                                         0x3de5d8fd1fcf0ec1ull>(z);
                return fma_scalar(y * z, x, x);
            }

            XSIMD_INLINE double base_tancot_eval_scalar(double z) noexcept
            {
                double zz = z * z;
                double num = horner_scalar<double,
                                           0xc1711fead3299176ull,
                                           0x413199eca5fc9dddull,
                                           0xc0c992d8d24f3f38ull>(zz);
                double den = horner1_scalar<double,
                                            0xc189afe03cbe5a31ull,
                                            0x4177d98fc2ead8efull,
                                            0xc13427bc582abc96ull,
                                            0x40cab8a5eeb36572ull>(zz);
                return fma_scalar(z, (zz * (num / den)), z);
            }
        }

        // ---- trigonometry ----

        template <class T>
        XSIMD_INLINE T sin(T self) noexcept
        {
            T x = detail::abs_scalar(self);
            T xr = detail::nan_like_batch<T>();
            T n = detail::trigo_reduce(x, xr);
            T tmp = n >= T(2) ? T(1) : T(0);
            T swap_bit = detail::fma_scalar(T(-2), tmp, n);
            T sign_bit = detail::flip_sign(detail::sign_bit_of(self),
                                           tmp != T(0) ? detail::sign_bit_of(T(-0.0)) : T(0));
            T z = xr * xr;
            T se = detail::sin_eval_scalar(z, xr);
            T ce = detail::cos_eval_scalar(z);
            T z1 = swap_bit == T(0) ? se : ce;
            return detail::flip_sign(z1, sign_bit);
        }

        template <class T>
        XSIMD_INLINE T cos(T self) noexcept
        {
            T x = detail::abs_scalar(self);
            T xr = detail::nan_like_batch<T>();
            T n = detail::trigo_reduce(x, xr);
            T tmp = n >= T(2) ? T(1) : T(0);
            T swap_bit = detail::fma_scalar(T(-2), tmp, n);
            T sign_bit = (swap_bit != tmp) ? detail::sign_bit_of(T(-0.0)) : T(0);
            T z = xr * xr;
            T se = detail::sin_eval_scalar(z, xr);
            T ce = detail::cos_eval_scalar(z);
            T z1 = swap_bit != T(0) ? se : ce;
            return detail::flip_sign(z1, sign_bit);
        }

        template <class T>
        XSIMD_INLINE T tan(T self) noexcept
        {
            T x = detail::abs_scalar(self);
            T xr = detail::nan_like_batch<T>();
            T n = detail::trigo_reduce(x, xr);
            T tmp = n >= T(2) ? T(1) : T(0);
            T swap_bit = detail::fma_scalar(T(-2), tmp, n);
            bool test = swap_bit == T(0);
            T y = detail::base_tancot_eval_scalar(xr);
            T r = test ? y : -T(1) / y;
            return detail::flip_sign(r, detail::sign_bit_of(self));
        }

        template <class T>
        XSIMD_INLINE T cot(T self) noexcept
        {
            // Defined as the exact reciprocal of the tan computation
            // (mirrors 1 / tan on the batch side, including the double
            // rounding of the !test branch, for bit-exactness).
            return T(1) / scalar::tan(self);
        }

        // ---- hyperbolic ----

        XSIMD_INLINE float exp(float x) noexcept;
        XSIMD_INLINE double exp(double x) noexcept;

        namespace detail
        {
            XSIMD_INLINE float sinh_kernel_scalar(float self) noexcept
            {
                float sqr_self = self * self;
                return horner_scalar<float,
                                     0x3f800000,
                                     0x3e2aaacc,
                                     0x3c087bbe,
                                     0x39559e2f>(sqr_self)
                    * self;
            }

            XSIMD_INLINE double sinh_kernel_scalar(double self) noexcept
            {
                double sqrself = self * self;
                return fma_scalar(self,
                                (horner_scalar<double,
                                               0xc115782bdbf6ab05ull,
                                               0xc0c694b8c71d6182ull,
                                               0xc064773a398ff4feull,
                                               0xbfe9435fe8bb3cd6ull>(sqrself)
                                     / horner1_scalar<double,
                                                      0xc1401a20e4f90044ull,
                                                      0x40e1a7ba7ed72245ull,
                                                      0xc0715b6096e96484ull>(sqrself))
                                    * sqrself,
                                self);
            }

            XSIMD_INLINE float tanh_kernel_scalar(float x) noexcept
            {
                float sqrx = x * x;
                return fma_scalar(horner_scalar<float,
                                               0xbeaaaa99,
                                               0x3e088393,
                                               0xbd5c1e2d,
                                               0x3ca9134e,
                                               0xbbbaf0ea>(sqrx)
                                    * sqrx,
                                x, x);
            }

            XSIMD_INLINE double tanh_kernel_scalar(double x) noexcept
            {
                double sqrx = x * x;
                // p/q rational form mirrors the double tanh kernel
                double p_ = horner_scalar<double,
                                          0xc0993ac030580563ull,
                                          0xc058d26a0e26682dull,
                                          0xbfeedc5baafd6f4b>(sqrx);
                double q_ = horner1_scalar<double,
                                           0x40b2ec102442040cull,
                                           0x40a176fa0e5535faull,
                                           0x405c33f28a581b86>(sqrx);
                return fma_scalar(sqrx * p_ / q_, x, x);
            }
        }

        template <class T>
        XSIMD_INLINE T sinh(T a) noexcept
        {
            T half = T(0.5);
            T x = detail::abs_scalar(a);
            bool lt1 = x < T(1);
            T bts = detail::sign_bit_of(a);
            if (lt1)
                return detail::flip_sign(detail::sinh_kernel_scalar(x), bts);
            bool test1 = x > (xsimd::constants::maxlog<T>() - xsimd::constants::log_2<T>());
            T fac = test1 ? half : T(1);
            T t = scalar::exp(x * fac);
            T t1 = half * t;
            T r = test1 ? t1 * t : t1 - half / t;
            return detail::flip_sign(r, bts);
        }

        template <class T>
        XSIMD_INLINE T cosh(T self) noexcept
        {
            T x = detail::abs_scalar(self);
            bool test1 = x > (xsimd::constants::maxlog<T>() - xsimd::constants::log_2<T>());
            T fac = test1 ? T(0.5) : T(1);
            T t = scalar::exp(x * fac);
            T t1 = T(0.5) * t;
            return test1 ? t1 * t : (t + T(1) / t) * T(0.5);
        }

        template <class T>
        XSIMD_INLINE T tanh(T self) noexcept
        {
            T x = detail::abs_scalar(self);
            bool test = x < (T(5) / T(8));
            T bts = detail::sign_bit_of(self);
            if (test)
                return detail::flip_sign(detail::tanh_kernel_scalar(x), bts);
            T r = detail::fma_scalar(T(-2), T(1) / (T(1) + scalar::exp(x + x)), T(1));
            return detail::flip_sign(r, bts);
        }

        template <class T>
        XSIMD_INLINE T coth(T self) noexcept
        {
            return T(1) / scalar::tanh(self);
        }

        // ---- exponential / logarithm ----

        namespace detail
        {
            // Mirror of exp_reduction<float, exp_tag>.
            XSIMD_INLINE float exp_reduce(float a, float& x) noexcept
            {
                float k = nearbyint_scalar(xsimd::constants::invlog_2<float>() * a);
                x = fma_scalar(-k, xsimd::constants::log_2hi<float>(), a);
                x = fma_scalar(-k, xsimd::constants::log_2lo<float>(), x);
                return k;
            }

            // Mirror of exp_reduction<double, exp_tag>.
            XSIMD_INLINE double exp_reduce(double a, double& hi, double& lo, double& x) noexcept
            {
                double k = nearbyint_scalar(xsimd::constants::invlog_2<double>() * a);
                hi = fma_scalar(-k, xsimd::constants::log_2hi<double>(), a);
                lo = k * xsimd::constants::log_2lo<double>();
                x = hi - lo;
                return k;
            }

            XSIMD_INLINE float exp_approx(float x) noexcept
            {
                float y = horner_scalar<float,
                                        0x3f000000,
                                        0x3e2aa9a5,
                                        0x3d2aa957,
                                        0x3c098d8b,
                                        0x3ab778cf>(x);
                return fma_scalar(y, x * x, x) + 1.f;
            }

            XSIMD_INLINE double exp_approx(double x) noexcept
            {
                double t = x * x;
                // mirrors fnma(t, horner, x) = -t * horner + x
                return fma_scalar(-t,
                                horner_scalar<double,
                                               0x3fc555555555553eull,
                                               0xbf66c16c16bebd93ull,
                                               0x3f11566aaf25de2cull,
                                               0xbebbbd41c5d26bf1ull,
                                               0x3e66376972bea4d0ull>(t),
                                x);
            }
        }

        XSIMD_INLINE float exp(float self) noexcept
        {
            float x;
            float k = detail::exp_reduce(self, x);
            x = detail::exp_approx(x);
            x = self <= xsimd::constants::minlog<float>() ? 0.f : detail::ldexp_scalar(x, static_cast<int32_t>(k));
#ifndef __FAST_MATH__
            x = self >= xsimd::constants::maxlog<float>() ? detail::inf_scalar(float()) : x;
#endif
            return x;
        }

        XSIMD_INLINE double exp(double self) noexcept
        {
            double hi, lo, x;
            double k = detail::exp_reduce(self, hi, lo, x);
            double c = detail::exp_approx(x);
            // finalize, mirrors exp_reduction<double, exp_tag>::finalize
            c = 1. - (((lo - (x * c) / (2. - c)) - hi));
            c = self <= xsimd::constants::minlog<double>() ? 0. : detail::ldexp_scalar(c, static_cast<int64_t>(k));
#ifndef __FAST_MATH__
            c = self >= xsimd::constants::maxlog<double>() ? detail::inf_scalar(double()) : c;
#endif
            return c;
        }

        XSIMD_INLINE float log(float self) noexcept
        {
            using i_type = int32_t;
            float x = self;
            i_type k = 0;
            bool isnez = self != 0.f;
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < xsimd::constants::smallestposval<float>()) && isnez;
            if (test)
            {
                k -= 23;
                x *= 8388608ul;
            }
#endif
            i_type ix = bit_cast<i_type>(x);
            ix += 0x3f800000 - 0x3f3504f3;
            k += (ix >> 23) - 0x7f;
            ix = (ix & 0x007fffff) + 0x3f3504f3;
            x = bit_cast<float>(ix);
            float f = x - 1.f;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * detail::horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * detail::horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t2 + t1;
            float hfsq = 0.5f * f * f;
            float dk = static_cast<float>(k);
            detail::scalar_reassociation_barrier(dk);
            float r = detail::fma_scalar(dk, xsimd::constants::log_2hi<float>(),
                                        detail::fma_scalar(s, (hfsq + R), dk * xsimd::constants::log_2lo<float>()) - hfsq + f);
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            float zz = isnez ? (self == detail::inf_scalar(float())
                                    ? detail::inf_scalar(float())
                                    : r)
                             : -detail::inf_scalar(float());
            return !(self >= 0.f) ? detail::nan_like_batch<float>() : zz;
#endif
        }

        XSIMD_INLINE double log(double self) noexcept
        {
            using i_type = int64_t;
            double x = self;
            i_type k = 0;
            bool isnez = self != 0.;
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < xsimd::constants::smallestposval<double>()) && isnez;
            if (test)
            {
                k -= 54;
                x *= 18014398509481984ull;
            }
#endif
            i_type hx_bits = bit_cast<i_type>(x);
            i_type hx = hx_bits >> 32;
            hx += 0x3ff00000 - 0x3fe6a09e;
            k += (hx >> 20) - 0x3ff;
            double dk = static_cast<double>(k);
            detail::scalar_reassociation_barrier(dk);
            hx = (hx & 0x000fffff) + 0x3fe6a09e;
            i_type low = hx_bits & 0xffffffff;
            x = bit_cast<double>((hx << 32) | low);

            double f = x - 1.;
            double hfsq = 0.5 * f * f;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;

            double t1 = w * detail::horner_scalar<double, 0x3fd999999997fa04ll, 0x3fcc71c51d8e78afll, 0x3fc39a09d078c69fll>(w);
            double t2 = z * detail::horner_scalar<double, 0x3fe5555555555593ll, 0x3fd2492494229359ll, 0x3fc7466496cb03dell, 0x3fc2f112df3e5244ll>(w);
            double R = t2 + t1;
            double r = detail::fma_scalar(dk, xsimd::constants::log_2hi<double>(),
                                           detail::fma_scalar(s, (hfsq + R), dk * xsimd::constants::log_2lo<double>()) - hfsq + f);
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            double zz = isnez ? (self == detail::inf_scalar(double())
                                     ? detail::inf_scalar(double())
                                     : r)
                              : -detail::inf_scalar(double());
            return !(self >= 0.) ? detail::nan_like_batch<double>() : zz;
#endif
        }

        float log1p(float self) noexcept;
        double log1p(double self) noexcept;

        /***********************************************************
         * Additional kernels: inverse trigonometric / hyperbolic,  *
         * exp2 / expm1 / log2 / log10 / log1p, mirroring the batch *
         * common kernels bit-for-bit.                              *
         ***********************************************************/
        namespace detail
        {
            XSIMD_INLINE float kernel_atan_scalar(float x, float recx) noexcept
            {
                using constants::pio2; using constants::pio4;
                using constants::pio_2lo; using constants::pio_4lo;
                using constants::tan3pio8;
                const bool flag1 = x < tan3pio8<float>();
                const bool flag2 = (x >= bit_cast<float>(0x3ed413cdu)) && flag1;
                float yy = flag1 ? 0.f : pio2<float>();
                yy = flag2 ? pio4<float>() : yy;
                float xx = flag1 ? x : -recx;
                xx = flag2 ? (x - 1.f) / (x + 1.f) : xx;
                const float z = xx * xx;
                float z1 = horner_scalar<float,
                                         0xbeaaaa2aul, 0x3e4c925ful,
                                         0xbe0e1b85ul, 0x3da4f0d1ul>(z);
                z1 = fma_scalar(xx, z1 * z, xx);
                z1 = flag2 ? z1 + pio_4lo<float>() : z1;
                z1 = !flag1 ? z1 + pio_2lo<float>() : z1;
                return yy + z1;
            }

            XSIMD_INLINE double kernel_atan_scalar(double x, double recx) noexcept
            {
                using constants::pio2; using constants::pio4;
                using constants::pio_2lo; using constants::pio_4lo;
                using constants::tan3pio8; using constants::tanpio8;
                const bool flag1 = x < tan3pio8<double>();
                const bool flag2 = (x >= tanpio8<double>()) && flag1;
                double yy = flag1 ? 0. : pio2<double>();
                yy = flag2 ? pio4<double>() : yy;
                double xx = flag1 ? x : -recx;
                xx = flag2 ? (x - 1.) / (x + 1.) : xx;
                double z = xx * xx;
                z *= horner_scalar<double,
                                   0xc0503669fd28ec8eull, 0xc05eb8bf2d05ba25ull,
                                   0xc052c08c36880273ull, 0xc03028545b6b807aull,
                                   0xbfec007fa1f72594ull>(z)
                    / horner1_scalar<double,
                                     0x4068519efbbd62ecull, 0x407e563f13b049eaull,
                                     0x407b0e18d2e2be3bull, 0x4064a0dd43b8fa25ull,
                                     0x4038dbc45b14603cull>(z);
                z = fma_scalar(xx, z, xx);
                z = flag2 ? z + pio_4lo<double>() : z;
                z = z + (flag1 ? 0. : pio_2lo<double>());
                return yy + z;
            }

            template <class T>
            XSIMD_INLINE T fms_scalar(T a, T b, T c) noexcept
            {
                return fma_scalar(a, b, -c);
            }

            template <class T>
            XSIMD_INLINE T fnma_scalar(T a, T b, T c) noexcept
            {
                return fma_scalar(-a, b, c);
            }

            template <class T>
            XSIMD_INLINE T averagef_scalar(T x1, T x2) noexcept
            {
                return fma_scalar(x1, T(0.5), x2 * T(0.5));
            }

            template <class T>
            XSIMD_INLINE T signnz_scalar(T v) noexcept { return v < T(0.) ? T(-1.) : T(1.); }

            // sqrt via the compiler builtin: lowers to the hardware sqrt
            // instruction (no libm call) and is bit-identical to the batch
            // sqrt, which every SIMD target also computes in hardware.
            template <class T>
            XSIMD_INLINE T sqrt_scalar(T x) noexcept
            {
                // __builtin_sqrt is double-typed; routing float through it
                // would add convert-round-trips, so dispatch on the type.
                if constexpr (sizeof(T) == 4)
                    return __builtin_sqrtf(x);
                else
                    return __builtin_sqrt(x);
            }

            template <class T>
            XSIMD_INLINE T hypot_scalar(T a, T b) noexcept
            {
                return sqrt_scalar(fma_scalar(a, a, b * b));
            }
        }

        // ---------------- asin ----------------
        XSIMD_INLINE float asin(float self) noexcept
        {
            using constants::pio2;
            float x = detail::abs_scalar(self);
            float sign = detail::sign_bit_of(self);
            bool x_larger_05 = x > 0.5f;
            float z = x_larger_05 ? 0.5f * (1.f - x) : x * x;
            x = x_larger_05 ? detail::sqrt_scalar(z) : x;
            float z1 = detail::horner_scalar<float,
                                             0x3e2aaae4, 0x3d9980f6, 0x3d3a3ec7,
                                             0x3cc617e3, 0x3d2cb352>(z);
            z1 = detail::fma_scalar(z1, z * x, x);
            if (z1 != z1) // NaN input propagates with the input sign restored,
                          // mirroring the batch float asin NaN handling
                return detail::flip_sign(z1, sign);
            z = x_larger_05 ? pio2<float>() - (z1 + z1) : z1;
            return sign != 0.f ? detail::flip_sign(z, sign) : z;
        }

        XSIMD_INLINE double asin(double self) noexcept
        {
            using constants::pio4; using constants::pio_2lo; using constants::sqrteps;
            double x = detail::abs_scalar(self);
            bool small_cond = x < sqrteps<double>();
            double ct1 = bit_cast<double>(int64_t(0x3fe4000000000000));
            double zz1 = 1. - x;
            double vp = zz1 * detail::horner_scalar<double, 0x403c896240f3081dull, 0xc03991aaac01ab68ull, 0x401bdff5baf33e6aull, 0xbfe2079259f9290full, 0x3f684fc3988e9f08ull>(zz1) / detail::horner1_scalar<double, 0x40756709b0b644beull, 0xc077fe08959063eeull, 0x40626219af6a7f42ull, 0xc035f2a2b6bf5d8cull>(zz1);
            zz1 = detail::sqrt_scalar(zz1 + zz1);
            double z = pio4<double>() - zz1;
            zz1 = detail::fms_scalar(zz1, vp, pio_2lo<double>());
            z = z - zz1;
            zz1 = z + pio4<double>();
            double zz2 = self * self;
            z = zz2 * detail::horner_scalar<double, 0xc020656c06ceafd5ull, 0x40339007da779259ull, 0xc0304331de27907bull, 0x4015c74b178a2dd9ull, 0xbfe34341333e5c16ull, 0x3f716b9b0bd48ad3ull>(zz2) / detail::horner1_scalar<double, 0xc04898220a3607acull, 0x4061705684ffbf9dull, 0xc06265bb6d3576d7ull, 0x40519fc025fe9054ull, 0xc02d7b590b5e0eabull>(zz2);
            zz2 = detail::fma_scalar(x, z, x);
            if (x > 1.)
                return detail::nan_like_batch<double>();
            if (x != x) // NaN input: bit-level abs, mirroring batch NaN propagation
                return x;
            double r = small_cond ? x : (x > ct1 ? zz1 : zz2);
            return detail::flip_sign(r, detail::sign_bit_of(self));
        }

        // ---------------- acos ----------------
        XSIMD_INLINE float acos(float self) noexcept
        {
            using constants::pi; using constants::pio2;
            float x = detail::abs_scalar(self);
            bool x_larger_05 = x > 0.5f;
            x = x_larger_05 ? detail::sqrt_scalar(detail::fma_scalar(-0.5f, x, 0.5f)) : self;
            x = asin(x);
            x = x_larger_05 ? x + x : x;
            x = self < -0.5f ? pi<float>() - x : x;
            return x_larger_05 ? x : pio2<float>() - x;
        }

        XSIMD_INLINE double acos(double self) noexcept
        {
            using constants::pi; using constants::pio2;
            double x = detail::abs_scalar(self);
            bool x_larger_05 = x > 0.5;
            x = x_larger_05 ? detail::sqrt_scalar(detail::fma_scalar(-0.5, x, 0.5)) : self;
            x = asin(x);
            x = x_larger_05 ? x + x : x;
            x = self < -0.5 ? pi<double>() - x : x;
            return x_larger_05 ? x : pio2<double>() - x;
        }

        // ---------------- atan / atan2 ----------------
        XSIMD_INLINE float atan(float self) noexcept
        {
            float absa = detail::abs_scalar(self);
            float x = detail::kernel_atan_scalar(absa, 1.f / absa);
            return detail::flip_sign(x, detail::sign_bit_of(self));
        }

        XSIMD_INLINE double atan(double self) noexcept
        {
            double absa = detail::abs_scalar(self);
            double x = detail::kernel_atan_scalar(absa, 1. / absa);
            return detail::flip_sign(x, detail::sign_bit_of(self));
        }

        XSIMD_INLINE float atan2(float self, float other) noexcept
        {
            using constants::pi;
            const float q = detail::abs_scalar(self / other);
            const float q_p = 1.f / q;
            const float z = detail::kernel_atan_scalar(q, q_p);
            return (other > 0.f ? z : pi<float>() - z) * detail::signnz_scalar(self);
        }

        XSIMD_INLINE double atan2(double self, double other) noexcept
        {
            using constants::pi;
            const double q = detail::abs_scalar(self / other);
            const double q_p = 1. / q;
            const double z = detail::kernel_atan_scalar(q, q_p);
            return (other > 0. ? z : pi<double>() - z) * detail::signnz_scalar(self);
        }

        // ---------------- asinh ----------------
        XSIMD_INLINE float asinh(float self) noexcept
        {
            using constants::log_2; using constants::oneosqrteps;
            float x = detail::abs_scalar(self);
            bool lthalf = x < 0.5f;
            float x2 = x * x;
            float bts = detail::sign_bit_of(self);
            float z = 0.f;
            if (lthalf)
            {
                z = detail::horner_scalar<float,
                                          0x3f800000, 0xbe2aa9ad, 0x3d9949b1,
                                          0xbd2ee581, 0x3ca4d6e6>(x2)
                    * x;
                return detail::flip_sign(z, bts);
            }
            float tmp = x > oneosqrteps<float>() ? x : detail::averagef_scalar(x, detail::hypot_scalar(1.f, x));
#if !defined(XSIMD_NO_NANS)
            if (x != x)
                return detail::nan_like_batch<float>();
#endif
            return detail::flip_sign(log(tmp) + log_2<float>(), bts);
        }

        XSIMD_INLINE double asinh(double self) noexcept
        {
            using constants::log_2; using constants::oneosqrteps;
            double x = detail::abs_scalar(self);
            bool test = x > oneosqrteps<double>();
            double z = test ? x - 1. : x + x * x / (1. + detail::hypot_scalar(1., x));
#if !defined(XSIMD_NO_INFINITIES)
            if (x == detail::inf_scalar(double()))
                z = x;
#endif
            double l1pz = log1p(z);
            z = test ? l1pz + log_2<double>() : l1pz;
            return detail::flip_sign(z, detail::sign_bit_of(self));
        }

        // ---------------- acosh ----------------
        XSIMD_INLINE float acosh(float self) noexcept
        {
            using constants::log_2; using constants::oneotwoeps;
            float x = self - 1.f;
            bool test = x > oneotwoeps<float>();
            float z = test ? self : x + detail::sqrt_scalar(x + x + x * x);
            float l1pz = log1p(z);
            return test ? l1pz + log_2<float>() : l1pz;
        }

        XSIMD_INLINE double acosh(double self) noexcept
        {
            using constants::log_2; using constants::oneotwoeps;
            double x = self - 1.;
            bool test = x > oneotwoeps<double>();
            double z = test ? self : x + detail::sqrt_scalar(x + x + x * x);
            double l1pz = log1p(z);
            return test ? l1pz + log_2<double>() : l1pz;
        }

        // ---------------- atanh ----------------
        XSIMD_INLINE float atanh(float self) noexcept
        {
            float x = detail::abs_scalar(self);
            float t = x + x;
            float z = 1.f - x;
            bool test = x < 0.5f;
            float tmp = (test ? x : t) / z;
            return detail::flip_sign(0.5f * log1p(test ? detail::fma_scalar(t, tmp, t) : tmp), detail::sign_bit_of(self));
        }

        XSIMD_INLINE double atanh(double self) noexcept
        {
            double x = detail::abs_scalar(self);
            double t = x + x;
            double z = 1. - x;
            bool test = x < 0.5;
            double tmp = (test ? x : t) / z;
            return detail::flip_sign(0.5 * log1p(test ? detail::fma_scalar(t, tmp, t) : tmp), detail::sign_bit_of(self));
        }

        // ---------------- exp2 ----------------
        XSIMD_INLINE float exp2(float self) noexcept
        {
            using constants::log_2;
            using constants::maxlog2; using constants::minlog2;
            float k = detail::nearbyint_scalar(self);
            float x = self - k;
            float y = detail::horner_scalar<float,
                                             0x3e75fdf1, 0x3d6356eb,
                                             0x3c1d9422, 0x3ab01218,
                                             0x3922c8c4>(x);
            x = 1.f + detail::fma_scalar(y, x * x, x * log_2<float>());
            x = self <= minlog2<float>() ? 0.f : detail::ldexp_scalar(x, static_cast<int32_t>(k));
#ifndef __FAST_MATH__
            if (self >= maxlog2<float>())
                return detail::inf_scalar(float());
#endif
            return x;
        }

        XSIMD_INLINE double exp2(double self) noexcept
        {
            using constants::log_2;
            using constants::maxlog2; using constants::minlog2;
            double k = detail::nearbyint_scalar(self);
            double x = (self - k) * log_2<double>();
            double t = x * x;
            double c = detail::fnma_scalar(t,
                                           detail::horner_scalar<double,
                                                                0x3fc555555555553eull,
                                                                0xbf66c16c16bebd93ull,
                                                                0x3f11566aaf25de2cull,
                                                                0xbebbbd41c5d26bf1ull,
                                                                0x3e66376972bea4d0ull>(t),
                                           x);
            c = 1. + x + x * c / (2. - c);
            c = self <= minlog2<double>() ? 0. : detail::ldexp_scalar(c, static_cast<int64_t>(k));
#ifndef __FAST_MATH__
            if (self >= maxlog2<double>())
                return detail::inf_scalar(double());
#endif
            return c;
        }

        // ---------------- expm1 ----------------
        XSIMD_INLINE float expm1(float self) noexcept
        {
            using constants::invlog_2; using constants::log_2hi; using constants::log_2lo;
            using constants::maxexponent; using constants::nmb;
            using constants::maxlog; using constants::logeps;
            float k = detail::nearbyint_scalar(invlog_2<float>() * self);
            float x = detail::fnma_scalar(k, log_2hi<float>(), self);
            x = detail::fnma_scalar(k, log_2lo<float>(), x);
            float hx = x * 0.5f;
            float hxs = x * hx;
            float r = detail::horner_scalar<float,
                                            0X3F800000UL, 0XBD08887FUL,
                                            0X3ACF6DB4UL>(hxs);
            float t = detail::fnma_scalar(r, hx, 3.f);
            float e = hxs * ((r - t) / (6.f - x * t));
            e = detail::fms_scalar(x, e, hxs);
            int32_t ik = static_cast<int32_t>(k);
            uint32_t bits_two2mk = static_cast<uint32_t>((maxexponent<float>() - ik) << nmb<float>());
            float two2mk = bit_cast<float>(bits_two2mk);
            float y = 1.f - two2mk - (e - x);
            y = detail::ldexp_scalar(y, ik);
            y = self >= maxlog<float>() ? detail::inf_scalar(float()) : y;
            return self < logeps<float>() ? -1.f : y;
        }

        XSIMD_INLINE double expm1(double self) noexcept
        {
            using constants::invlog_2; using constants::log_2hi; using constants::log_2lo;
            using constants::maxexponent; using constants::nmb;
            using constants::maxlog; using constants::logeps;
            double k = detail::nearbyint_scalar(invlog_2<double>() * self);
            double hi = detail::fnma_scalar(k, log_2hi<double>(), self);
            double lo = k * log_2lo<double>();
            double x = hi - lo;
            double hxs = x * x * 0.5;
            double r = detail::horner_scalar<double,
                                             0X3FF0000000000000ULL,
                                             0XBFA11111111110F4ULL,
                                             0X3F5A01A019FE5585ULL,
                                             0XBF14CE199EAADBB7ULL,
                                             0X3ED0CFCA86E65239ULL,
                                             0XBE8AFDB76E09C32DULL>(hxs);
            double t = 3. - r * 0.5 * x;
            double e = hxs * ((r - t) / (6. - x * t));
            double c = (hi - x) - lo;
            e = (x * (e - c) - c) - hxs;
            int64_t ik = static_cast<int64_t>(k);
            uint64_t bits_two2mk = static_cast<uint64_t>(maxexponent<double>() - ik) << nmb<double>();
            double two2mk = bit_cast<double>(bits_two2mk);
            double ct1 = 1. - two2mk - (e - x);
            double ct2 = 1. + (x - (e + two2mk));
            double y = k < 20. ? ct1 : ct2;
            y = detail::ldexp_scalar(y, ik);
#ifndef __FAST_MATH__
            y = self >= maxlog<double>() ? detail::inf_scalar(double()) : y;
#endif
            return self < logeps<double>() ? -1. : y;
        }

        // ---------------- log1p ----------------
        XSIMD_INLINE float log1p(float self) noexcept
        {
            using constants::log_2hi; using constants::log_2lo;
            using i_type = int32_t;
            const float uf = self + 1.f;
            bool isnez = uf != 0.f;
            i_type iu = bit_cast<i_type>(uf);
            iu += 0x3f800000 - 0x3f3504f3;
            i_type k = (iu >> 23) - 0x7f;
            iu = (iu & 0x007fffff) + 0x3f3504f3;
            float f = bit_cast<float>(iu) - 1.f;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * detail::horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * detail::horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t2 + t1;
            float hfsq = 0.5f * f * f;
            float dk = static_cast<float>(k);
            detail::scalar_reassociation_barrier(dk);
            /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
            float c = (k >= 2 ? 1.f - (uf - self) : self - (uf - 1.f)) / uf;
            float r = detail::fma_scalar(dk, log_2hi<float>(),
                                         detail::fma_scalar(s, (hfsq + R), dk * log_2lo<float>() + c) - hfsq + f);
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            float zz = isnez ? (self == detail::inf_scalar(float())
                                    ? detail::inf_scalar(float())
                                    : r)
                             : -detail::inf_scalar(float());
            return !(uf >= 0.f) ? detail::nan_like_batch<float>() : zz;
#endif
        }

        XSIMD_INLINE double log1p(double self) noexcept
        {
            using constants::log_2hi; using constants::log_2lo;
            using i_type = int64_t;
            const double uf = self + 1.;
            bool isnez = uf != 0.;
            i_type hu = bit_cast<i_type>(uf) >> 32;
            hu += 0x3ff00000 - 0x3fe6a09e;
            i_type k = (hu >> 20) - 0x3ff;
            /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
            double c = (k >= 2 ? 1. - (uf - self) : self - (uf - 1.)) / uf;
            hu = (hu & 0x000fffff) + 0x3fe6a09e;
            double f = bit_cast<double>((hu << 32) | (i_type(0xffffffff) & bit_cast<i_type>(uf)));
            f = f - 1.;
            double hfsq = 0.5 * f * f;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;
            double t1 = w * detail::horner_scalar<double, 0x3fd999999997fa04ll, 0x3fcc71c51d8e78afll, 0x3fc39a09d078c69fll>(w);
            double t2 = z * detail::horner_scalar<double, 0x3fe5555555555593ll, 0x3fd2492494229359ll, 0x3fc7466496cb03dell, 0x3fc2f112df3e5244ll>(w);
            double R = t2 + t1;
            double dk = static_cast<double>(k);
            detail::scalar_reassociation_barrier(dk);
            double r = detail::fma_scalar(dk, log_2hi<double>(),
                                          detail::fma_scalar(s, hfsq + R, dk * log_2lo<double>() + c) - hfsq + f);
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            double zz = isnez ? (self == detail::inf_scalar(double())
                                     ? detail::inf_scalar(double())
                                     : r)
                              : -detail::inf_scalar(double());
            return !(uf >= 0.) ? detail::nan_like_batch<double>() : zz;
#endif
        }

        // ---------------- log2 ----------------
        XSIMD_INLINE float log2(float self) noexcept
        {
            using constants::invlog_2;
            using i_type = int32_t;
            using constants::smallestposval;
            float x = self;
            i_type k(0);
            bool isnez = (self != 0.f);
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < smallestposval<float>()) && isnez;
            if (test)
            {
                k -= 25;
                x *= 33554432ul;
            }
#endif
            i_type ix = bit_cast<i_type>(x);
            ix += 0x3f800000 - 0x3f3504f3;
            k += (ix >> 23) - 0x7f;
            ix = (ix & 0x007fffff) + 0x3f3504f3;
            x = bit_cast<float>(ix);
            float f = x - 1.f;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * detail::horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * detail::horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t1 + t2;
            float hfsq = 0.5f * f * f;
            float dk = static_cast<float>(k);
            detail::scalar_reassociation_barrier(dk);
            float r = detail::fma_scalar(f + detail::fms_scalar(s, hfsq + R, hfsq), invlog_2<float>(), dk);
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            float zz = isnez ? (self == detail::inf_scalar(float())
                                    ? detail::inf_scalar(float())
                                    : r)
                             : -detail::inf_scalar(float());
            return !(self >= 0.f) ? detail::nan_like_batch<float>() : zz;
#endif
        }

        XSIMD_INLINE double log2(double self) noexcept
        {
            using constants::invlog_2hi; using constants::invlog_2lo;
            using i_type = int64_t;
            using constants::smallestposval;
            double x = self;
            i_type hx = bit_cast<i_type>(x) >> 32;
            i_type k(0);
            bool isnez = (self != 0.);
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < smallestposval<double>()) && isnez;
            if (test)
            {
                k -= 54;
                x *= 18014398509481984ull;
            }
#endif
            hx += 0x3ff00000 - 0x3fe6a09e;
            k += (hx >> 20) - 0x3ff;
            hx = (hx & 0x000fffff) + 0x3fe6a09e;
            x = bit_cast<double>(hx << 32 | (i_type(0xffffffff) & bit_cast<i_type>(x)));
            double f = x - 1.;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;
            double t1 = w * detail::horner_scalar<double, 0x3fd999999997fa04ll, 0x3fcc71c51d8e78afll, 0x3fc39a09d078c69fll>(w);
            double t2 = z * detail::horner_scalar<double, 0x3fe5555555555593ll, 0x3fd2492494229359ll, 0x3fc7466496cb03dell, 0x3fc2f112df3e5244ll>(w);
            double R = t2 + t1;
            double hfsq = 0.5 * f * f;
            double hi = f - hfsq;
            // truncate hi to 32 significant bits, mirroring the batch bit-and
            hi = bit_cast<double>(bit_cast<int64_t>(hi) & 0xffffffff00000000ll);
            double lo = detail::fma_scalar(s, hfsq + R, f - hi - hfsq);
            double dk = static_cast<double>(k);
            detail::scalar_reassociation_barrier(dk);
            double val_hi = hi * invlog_2hi<double>();
            double val_lo = detail::fma_scalar(lo + hi, invlog_2lo<double>(), lo * invlog_2hi<double>());
            double w1 = dk + val_hi;
            val_lo += (dk - w1) + val_hi;
            val_hi = w1;
            double r = val_lo + val_hi;
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            double zz = isnez ? (self == detail::inf_scalar(double())
                                     ? detail::inf_scalar(double())
                                     : r)
                              : -detail::inf_scalar(double());
            return !(self >= 0.) ? detail::nan_like_batch<double>() : zz;
#endif
        }

        // ---------------- log10 ----------------
        XSIMD_INLINE float log10(float self) noexcept
        {
            using i_type = int32_t;
            using constants::smallestposval;
            const float ivln10hi = 4.3432617188e-01f;
            const float ivln10lo = -3.1689971365e-05f;
            const float log10_2hi = 3.0102920532e-01f;
            const float log10_2lo = 7.9034151668e-07f;
            float x = self;
            i_type k(0);
            bool isnez = (self != 0.f);
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < smallestposval<float>()) && isnez;
            if (test)
            {
                k -= 25;
                x *= 33554432ul;
            }
#endif
            i_type ix = bit_cast<i_type>(x);
            ix += 0x3f800000 - 0x3f3504f3;
            k += (ix >> 23) - 0x7f;
            ix = (ix & 0x007fffff) + 0x3f3504f3;
            x = bit_cast<float>(ix);
            float f = x - 1.f;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * detail::horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * detail::horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t2 + t1;
            float dk = static_cast<float>(k);
            detail::scalar_reassociation_barrier(dk);
            float hfsq = 0.5f * f * f;
            float hibits = f - hfsq;
            hibits = bit_cast<float>(bit_cast<int32_t>(hibits) & 0xfffff000);
            float lobits = detail::fma_scalar(s, hfsq + R, f - hibits - hfsq);
            float r = detail::fma_scalar(dk, log10_2hi,
                                         detail::fma_scalar(hibits, ivln10hi,
                                                            detail::fma_scalar(lobits, ivln10hi,
                                                                               detail::fma_scalar(lobits + hibits, ivln10lo, dk * log10_2lo))));
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            float zz = isnez ? (self == detail::inf_scalar(float())
                                    ? detail::inf_scalar(float())
                                    : r)
                             : -detail::inf_scalar(float());
            return !(self >= 0.f) ? detail::nan_like_batch<float>() : zz;
#endif
        }

        XSIMD_INLINE double log10(double self) noexcept
        {
            using i_type = int64_t;
            using constants::smallestposval;
            const double ivln10hi = 4.34294481878168880939e-01;
            const double ivln10lo = 2.50829467116452752298e-11;
            const double log10_2hi = 3.01029995663611771306e-01;
            const double log10_2lo = 3.69423907715893078616e-13;
            double x = self;
            i_type hx = bit_cast<i_type>(x) >> 32;
            i_type k(0);
            bool isnez = (self != 0.);
#ifndef XSIMD_NO_DENORMALS
            bool test = (self < smallestposval<double>()) && isnez;
            if (test)
            {
                k -= 54;
                x *= 18014398509481984ull;
            }
#endif
            hx += 0x3ff00000 - 0x3fe6a09e;
            k += (hx >> 20) - 0x3ff;
            double dk = static_cast<double>(k);
            detail::scalar_reassociation_barrier(dk);
            hx = (hx & 0x000fffff) + 0x3fe6a09e;
            x = bit_cast<double>(hx << 32 | (i_type(0xffffffff) & bit_cast<i_type>(x)));
            double f = x - 1.;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;
            double t1 = w * detail::horner_scalar<double, 0x3fd999999997fa04ll, 0x3fcc71c51d8e78afll, 0x3fc39a09d078c69fll>(w);
            double t2 = z * detail::horner_scalar<double, 0x3fe5555555555593ll, 0x3fd2492494229359ll, 0x3fc7466496cb03dell, 0x3fc2f112df3e5244ll>(w);
            double R = t2 + t1;
            double hfsq = 0.5 * f * f;
            double hi = f - hfsq;
            hi = bit_cast<double>(bit_cast<int64_t>(hi) & 0xffffffff00000000ll);
            double lo = f - hi - hfsq + s * (hfsq + R);
            double val_hi = hi * ivln10hi;
            double y = dk * log10_2hi;
            double val_lo = dk * log10_2lo + (lo + hi) * ivln10lo + lo * ivln10hi;
            double w1 = y + val_hi;
            val_lo += (y - w1) + val_hi;
            val_hi = w1;
            double r = val_lo + val_hi;
#if defined(__FAST_MATH__) || defined(XSIMD_SCALAR_SKIP_SPECIALS)
            return r;
#else
            double zz = isnez ? (self == detail::inf_scalar(double())
                                     ? detail::inf_scalar(double())
                                     : r)
                              : -detail::inf_scalar(double());
            return !(self >= 0.) ? detail::nan_like_batch<double>() : zz;
#endif
        }

        /***********************************************************
         * Misc kernels replacing the std:: routing: bit-level or  *
         * branch-only implementations, no libm.                   *
         ***********************************************************/

        // copysign: take magnitude of x, sign of y.
        template <class T>
        XSIMD_INLINE T copysign(T x, T y) noexcept
        {
            return detail::flip_sign(detail::abs_scalar(x), detail::sign_bit_of(y));
        }

        // fdim: max(x - y, 0)
        template <class T>
        XSIMD_INLINE T fdim(T x, T y) noexcept
        {
            return x > y ? x - y : T(0.);
        }

        // fmax / fmin: NaN handling mirrors the batch select semantics
        // (comparison with NaN is false, so the other operand wins).
        template <class T>
        XSIMD_INLINE T fmax(T x, T y) noexcept
        {
            return x > y ? x : (x <= y ? y : x);
        }

        template <class T>
        XSIMD_INLINE T fmin(T x, T y) noexcept
        {
            return x < y ? x : (x >= y ? y : x);
        }

        // fmod: remainder of x/y with the sign of x, computed by
        // repeated magnitude reduction (libm-free).
        template <class T>
        XSIMD_INLINE T fmod(T x, T y) noexcept
        {
            if (y == T(0.) || detail::abs_scalar(x) == detail::inf_scalar(T())
                || detail::abs_scalar(y) == detail::inf_scalar(T()) && x != detail::inf_scalar(T()))
                return detail::nan_like_batch<T>();
            T r = x - detail::trunc_scalar(x / y) * y;
            // one correction step: the truncated quotient may be off by one
            if (detail::abs_scalar(r) >= detail::abs_scalar(y))
                r -= (r < T(0.)) == (y < T(0.)) ? y : -y;
            if (r == T(0.))
                r = detail::flip_sign(r, detail::sign_bit_of(x));
            return r;
        }

        // remainder: result has the sign of x, |r| <= |y|/2
        template <class T>
        XSIMD_INLINE T remainder(T x, T y) noexcept
        {
            T q = x / y;
            T n = detail::nearbyint_scalar(q);
            if (detail::abs_scalar(q - n) == T(0.5))
                n = (n > T(0.)) ? n + T(0.5) : n - T(0.5);  // round-half-to-even quotient
            T r = x - n * y;
            return r;
        }

        // rounding family: thin public aliases over the bit-level
        // mirrors of the batch rounding kernels (trunc/floor/ceil/
        // nearbyint live in detail, round is implemented here).
        template <class T>
        XSIMD_INLINE T hypot(T x, T y) noexcept
        {
            return detail::hypot_scalar(x, y);
        }

        template <class T>
        XSIMD_INLINE T round(T x) noexcept
        {
            T v = detail::abs_scalar(x);
            T c = detail::ceil_scalar(v);
            T cp = c - T(0.5) > v ? c - T(1.) : c;
            return v > xsimd::constants::twotonmb<T>() ? x : detail::flip_sign(cp, detail::sign_bit_of(x));
        }

        // modf: split into integer and fractional part, both with the
        // sign of x.
        template <class T>
        XSIMD_INLINE T modf(T x, T* iptr) noexcept
        {
            T i = detail::trunc_scalar(x);
            *iptr = i;
            return x - i;
        }

        // cbrt: scale the argument to [1, 8) with a power-of-two exponent,
        // then Newton iterations; the scaling keeps the iteration
        // well-conditioned for any input magnitude (libm-free).
        template <class T>
        XSIMD_INLINE T cbrt(T x) noexcept
        {
            if (x != x)
                return x;
            bool neg = x < T(0.);
            T a = neg ? -x : x;
            if (a == T(0.) || a == detail::inf_scalar(T()))
                return x;
            // exponent decomposition without frexp: exponent as integer
            as_integer_t<T> ib = bit_cast<as_integer_t<T>>(a);
            int e = int((ib >> xsimd::constants::nmb<T>()) - as_integer_t<T>(xsimd::constants::maxexponent<T>()));
            int e3 = e / 3;
            T scaled = detail::ldexp_scalar(a, -3 * e3); // in [1, 8)
            // coarse initial guess on [1, 8); Newton converges from any
            // positive start
            T r = scaled < T(2.) ? T(1.2) : (scaled < T(4.) ? T(2.) : T(3.));
            for (int k = 0; k < 8; ++k)
                r = (T(2.) * r + scaled / (r * r)) / T(3.);
            return detail::ldexp_scalar(r, e3) * (neg ? T(-1.) : T(1.));
        }

        /***********************************************************
         * Error and gamma functions, mirroring the batch common    *
         * kernels (boost/simd origin) without libm.                *
         ***********************************************************/

        namespace detail
        {
            // sin(z * pi): trigo_pi_tag reducer (xi = nearbyint(x*2),
            // xr = (x - xi/2) * pi) followed by the sin/cos evaluations
            // and the swap/sign reconstruction of kernel::detail::sin.
            template <class T>
            XSIMD_INLINE T sin_pi(T self) noexcept
            {
                T x = detail::abs_scalar(self);
                T xi = detail::nearbyint_scalar(x * T(2.));
                T x2 = x - xi * T(0.5);
                T xr = x2 * xsimd::constants::pi<T>();
                T n = detail::quadrant_scalar(xi);
                T tmp = n >= T(2.) ? T(1.) : T(0.);
                T swap_bit = detail::fma_scalar(T(-2.), tmp, n);
                T sign_bit = detail::flip_sign(detail::sign_bit_of(self), tmp != T(0.) ? detail::sign_bit_of(T(-0.)) : T(0.));
                T z = xr * xr;
                T se = detail::sin_eval_scalar(z, xr);
                T ce = detail::cos_eval_scalar(z);
                T z1 = swap_bit == T(0.) ? se : ce;
                return detail::flip_sign(z1, sign_bit);
            }

            // erf kernels, mirroring erf_kernel<batch<float>> /
            // erf_kernel<batch<double>>.
            XSIMD_INLINE float erf1_float(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3f906eba, 0xbec0937e, 0x3de70f22,
                                              0xbcdb61f4, 0x3ba4468d, 0xba1fc83b>(x);
            }

            XSIMD_INLINE float erfc2_float(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3f0a0e8b, 0xbf918a62, 0x3e243828, 0x3ec4ca6e,
                                              0x3e1175c7, 0x3e2006f0, 0xbfaea865, 0x4050b063,
                                              0xc0cd1a85, 0x40d67e3b, 0xc0283611>(x);
            }

            XSIMD_INLINE float erfc3_float(float x) noexcept
            {
                return (1.f - x) * detail::horner_scalar<float,
                                                         0x3f7ffffe, 0xbe036d7e, 0xbfa11698,
                                                         0xbffc9284, 0xc016c985, 0x3f2cff3b,
                                                         0xc010d956, 0x401b5680, 0x41aa8e55>(x);
            }

            XSIMD_INLINE double erf1_double(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3ff20dd750429b61ull, 0x3fc16500f106c0a5ull,
                                              0x3fa4a59a4f02579cull, 0x3f53b7664358865aull,
                                              0x3f110512d5b20332ull>(x)
                    / detail::horner_scalar<double,
                                            0x3ff0000000000000ull, 0x3fdd0a84eb1ca867ull,
                                            0x3fb64536ca92ea2full, 0x3f8166f75999dbd1ull,
                                            0x3f37ea4332348252ull>(x);
            }

            XSIMD_INLINE double erfc2_double(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3feffffffbbb552bull, 0x3ff54dfe9b258a60ull,
                                              0x3fec1986509e687bull, 0x3fd53dd7a67c7e9full,
                                              0x3fb2488a6b5cb5e5ull, 0x3f7cf4cfe0aacbb4ull,
                                              0x0ull>(x)
                    / detail::horner_scalar<double,
                                            0x3ff0000000000000ull, 0x4003adeae79b9708ull,
                                            0x40053b1052dca8bdull, 0x3ff9e677c2777c3cull,
                                            0x3fe307622fcff772ull, 0x3fc033c113a7deeeull,
                                            0x3f89a996639b0d00ull>(x);
            }

            XSIMD_INLINE double erfc3_double(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3fefff5a9e697ae2ull, 0x3ff9fa202deb88e5ull,
                                              0x3ff44744306832aeull, 0x3fe29be1cff90d94ull,
                                              0x3fc42210f88b9d43ull, 0x3f971d0907ea7a92ull,
                                              0x0ull>(x)
                    / detail::horner_scalar<double,
                                            0x3ff0000000000000ull, 0x400602f24bf3fdb6ull,
                                            0x400afd487397568full, 0x400315ffdfd5ce91ull,
                                            0x3ff0cfd4cb6cde9full, 0x3fd1d7ab774bb837ull,
                                            0x3fa47bd61bbb3843ull>(x);
            }

            // stirling kernel, mirroring stirling_kernel / stirling.
            template <class T>
            XSIMD_INLINE T stirling_compute(T w) noexcept;

            template <>
            XSIMD_INLINE float stirling_compute<float>(float w) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3daaaaab, 0x3b638e39, 0xbb2fb930, 0xb970b359>(w);
            }

            template <>
            XSIMD_INLINE double stirling_compute<double>(double w) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3fb5555555555986ull, 0x3f6c71c71b98c5fdull,
                                              0xbf65f72607d44fd7ull, 0xbf2e166b27e61d7cull,
                                              0x3f49cc72592d7293ull>(w);
            }

            template <class T>
            XSIMD_INLINE T stirling(T a) noexcept
            {
                T split_lim = bit_cast<T>(sizeof(T) == 4 ? as_unsigned_integer_t<T>(0x41d628f6u) : as_unsigned_integer_t<T>(0x4061e083ba3443d4ull));
                T large_lim = bit_cast<T>(sizeof(T) == 4 ? as_unsigned_integer_t<T>(0x420c28f3u) : as_unsigned_integer_t<T>(0x4065800000000000ull));
                T x = a >= T(0.) ? a : detail::nan_like_batch<T>();
                T w = T(1.) / x;
                w = detail::fma_scalar(w, detail::stirling_compute(w), T(1.));
                T y = exp(-x);
                bool test = x < split_lim;
                T z = x - T(0.5);
                z = test ? z : T(0.5) * z;
                T v = exp(z * log(detail::abs_scalar(x)));
                y *= v;
                y = test ? y : y * v;
                y *= xsimd::constants::sqrt_2pi<T>() * w;
#ifndef __FAST_MATH__
                y = (x == detail::inf_scalar(T()) || x == -detail::inf_scalar(T())) ? x : y;
                if (x > large_lim)
                    return detail::inf_scalar(T());
#endif
                return y;
            }

            // tgamma kernel, mirroring tgamma_kernel::compute.
            XSIMD_INLINE float tgamma_kernel_float(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3f800000UL, 0x3ed87799UL, 0x3ed2d411UL, 0x3da82a34UL,
                                              0x3d93ae7cUL, 0x3b91db14UL, 0x3ba90c99UL, 0x3ad28b22UL>(x);
            }

            XSIMD_INLINE double tgamma_kernel_double(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3ff0000000000000ULL, 0x3fdfa1373993e312ULL,
                                              0x3fca8da9dcae7d31ULL, 0x3fa863d918c423d3ULL,
                                              0x3f8557cde9db14b0ULL, 0x3f5384e3e686bfabULL,
                                              0x3f24fcb839982153ULL>(x)
                    / detail::horner_scalar<double,
                                            0x3ff0000000000000ULL, 0x3fb24944c9cd3c51ULL,
                                            0xbfce071a9d4287c2ULL, 0x3fa25779e33fde67ULL,
                                            0x3f8831ed5b1bb117ULL, 0xBf7240e4e750b44aULL,
                                            0x3f41ae8a29152573ULL, 0xbef8487a8400d3aFULL>(x);
            }

            template <class T>
            XSIMD_INLINE bool is_flint(T x) noexcept
            {
                return detail::trunc_scalar(x) == x;
            }

            // odd integer test on the truncated value (batch is_odd
            // semantics: works on flints, t - 2*trunc(t/2) != 0)
            template <class T>
            XSIMD_INLINE bool is_odd_scalar(T x) noexcept
            {
                T t = detail::trunc_scalar(x);
                T h = detail::trunc_scalar(t * T(0.5));
                return t - T(2.) * h != T(0.);
            }

            template <class T>
            XSIMD_INLINE T tgamma_large_negative(T a) noexcept
            {
                T st = detail::stirling(a);
                T p = detail::floor_scalar(a);
                T sgngam = detail::is_flint(p / T(2.)) ? T(-1.) : T(1.);
                T z = a - p;
                if (z < T(0.5))
                    z = z - T(1.);
                z = a * detail::sin_pi(z);
                z = detail::abs_scalar(z);
                return sgngam * xsimd::constants::pi<T>() / (z * st);
            }

            template <class T>
            XSIMD_INLINE T tgamma_other(T a) noexcept
            {
                T x = a;
                bool inf_result = false;
#ifndef XSIMD_NO_INFINITIES
                inf_result = (a == detail::inf_scalar(T()));
                if (inf_result)
                    x = T(2.);
#endif
                T z = T(1.);
                while (x >= T(3.))
                {
                    x = x - T(1.);
                    z = z * x;
                }
                while (x < T(0.))
                {
                    z = z / x;
                    x = x + T(1.);
                }
                while (x < T(2.))
                {
                    z = z / x;
                    x = x + T(1.);
                }
                x = z * (sizeof(T) == 4 ? detail::tgamma_kernel_float(x - T(2.))
                                        : detail::tgamma_kernel_double(x - T(2.)));
#ifndef XSIMD_NO_INFINITIES
                return inf_result ? a : x;
#else
                return x;
#endif
            }

            // lgamma kernels, mirroring gammalnA/B/C/1/2 and lgamma_impl.
            XSIMD_INLINE float gammalnB(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3ed87730, 0x3ea51a64, 0xbd89f07e, 0x3ca89ed8,
                                              0xbbf164fd, 0x3b3ba883, 0xbaabeab1, 0x3a1ebb94>(x);
            }

            XSIMD_INLINE float gammalnC(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0xbf13c468, 0x3f528d34, 0xbecd27a8, 0x3e8a898b,
                                              0xbe53c04f, 0x3e2d4dab, 0xbe22d329, 0x3e0c3c4f>(x);
            }

            XSIMD_INLINE float gammaln2(float x) noexcept
            {
                return detail::horner_scalar<float,
                                              0x3daaaa94, 0xbb358701, 0x3a31fd69>(x);
            }

            XSIMD_INLINE double gammaln1(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0xc12a0c675418055eull, 0xc13a45890219f20bull,
                                              0xc131bc82f994db51ull, 0xc1143d73f89089e5ull,
                                              0xc0e2f234355bb93eull, 0xc09589018ff36761ull>(x)
                    / detail::horner_scalar<double,
                                            0xc13ece4b6a11e14aull, 0xc1435255892ff34cull,
                                            0xc131628671950043ull, 0xc10aeb84b9744c9bull,
                                            0xc0d0aa0d7b89d757ull, 0xc075fd0d1cf312b2ull,
                                            0x3ff0000000000000ull>(x);
            }

            XSIMD_INLINE double gammalnA(double x) noexcept
            {
                return detail::horner_scalar<double,
                                              0x3fb555555555554bull, 0xbf66c16c16b0a5a1ull,
                                              0x3f4a019f20dc5ebbull, 0xbf437fbdb580e943ull,
                                              0x3f4a985027336661ull>(x);
            }

            XSIMD_INLINE float lgamma_negative(float q, float w) noexcept
            {
                float p = detail::floor_scalar(q);
                float z = q - p;
                if (z < 0.5f)
                    z = z - 1.f;
                z = q * detail::sin_pi(z);
                return -log(xsimd::constants::invpi<float>() * detail::abs_scalar(z)) - w;
            }

            XSIMD_INLINE float lgamma_other_float(float x) noexcept
            {
                bool xlt650 = x < 6.5f;
                float r0x = x, r0z = x, r0s = 1.f, r1 = 0.f;
                float p = detail::nan_like_batch<float>();
                if (xlt650)
                {
                    float z = 1.f;
                    float tx = x;
                    float nx = 0.f;
                    const float _075 = 0.75f, _150 = 1.50f, _125 = 1.25f, _250 = 2.50f;
                    bool xge150 = x >= _150;

                    while (xge150 && tx > _250)
                    {
                        nx = nx - 1.f;
                        tx = x + nx;
                        z = z * tx;
                    }
                    if (xge150)
                    {
                        r0x = x + nx - 2.f;
                        r0z = z;
                        r0s = 1.f;
                    }

                    bool xge125 = x >= _125;
                    if (xge125 && !xge150)
                    {
                        r0x = x - 1.f;
                        r0z = z * x;
                        r0s = -1.f;
                    }

                    bool kernelC = false;
                    bool xge075 = x >= _075;
                    if (xge075 && !xge125)
                    {
                        kernelC = true;
                        r0x = x - 1.f;
                        r0z = 1.f;
                        r0s = -1.f;
                        p = detail::gammalnC(r0x);
                    }

                    if (tx < _150 && !xge075)
                    {
                        while (tx < _150 && !xge075)
                        {
                            z = z * tx;
                            nx = nx + 1.f;
                            tx = x + nx;
                        }
                        r0x = r0x + nx - 2.f;
                        r0z = z;
                        r0s = -1.f;
                    }
                    p = kernelC ? p : detail::gammalnB(r0x);
                    // all lanes took the small-argument path
                    return detail::fma_scalar(r0x, p, r0s * log(detail::abs_scalar(r0z)));
                }
                // x >= 6.5: Stirling-type asymptotic (mirror of the batch
                // select(xlt650, r1, r2) tail)
                float m = log(x);
                return detail::fma_scalar(x - 0.5f, m, xsimd::constants::logsqrt2pi<float>() - x)
                    + detail::gammaln2(1.f / (x * x)) / x;
            }

            XSIMD_INLINE double lgamma_large_negative_double(double q) noexcept;

            XSIMD_INLINE double lgamma_other_double(double xx) noexcept
            {
                double x = xx;
                bool test = x < 13.;
                double r1 = 0.;
                if (test)
                {
                    double z = 1.;
                    double p = 0.;
                    double u = x;
                    while (u >= 3.)
                    {
                        p = p - 1.;
                        u = x + p;
                        z = z * u;
                    }
                    while (u < 2.)
                    {
                        z = z / u;
                        p = p + 1.;
                        u = x + p;
                    }
                    z = detail::abs_scalar(z);
                    x += p - 2.;
                    r1 = x * detail::gammaln1(x) + log(z);
                    return r1;
                }
                double r2 = detail::fma_scalar(xx - 0.5, log(xx), xsimd::constants::logsqrt2pi<double>() - xx);
                double p = 1. / (xx * xx);
                r2 += detail::gammalnA(p) / xx;
                return r2;
            }
        }

        XSIMD_INLINE float erf(float self) noexcept
        {
            float x = detail::abs_scalar(self);
            float r1 = 0.f;
            bool test1 = x < 2.f / 3.f;
            if (test1)
                return self * detail::erf1_float(x * x);
            float z = x / (1.f + x);
            z -= 0.4f;
            float r2 = 1.f - exp(-x * x) * detail::erfc2_float(z);
            r2 = self < 0.f ? -r2 : r2;
#ifndef XSIMD_NO_INFINITIES
            if (self == detail::inf_scalar(float()) || self == -detail::inf_scalar(float()))
                return self < 0.f ? -1.f : 1.f;
#endif
            return r2;
        }

        XSIMD_INLINE double erf(double self) noexcept
        {
            double x = detail::abs_scalar(self);
            double xx = x * x;
            const double lim1 = 0.65, lim2 = 2.2;
            bool test1 = x < lim1;
            if (test1)
                return self * detail::erf1_double(xx);
            bool test2 = x < lim2;
            double ex = exp(-xx);
            if (test2)
            {
                double z = 1. - ex * detail::erfc2_double(x);
                return self < 0. ? -z : z;
            }
            double z = 1. - ex * detail::erfc3_double(x);
            z = self < 0. ? -z : z;
#ifndef XSIMD_NO_INFINITIES
            if (self == detail::inf_scalar(double()) || self == -detail::inf_scalar(double()))
                return self < 0. ? -1. : 1.;
#endif
            return z;
        }

        XSIMD_INLINE float erfc(float self) noexcept
        {
            float x = detail::abs_scalar(self);
            bool test0 = self < 0.f;
            bool test1 = 3.f * x < 2.f;
            float z = x / (1.f + x);
            float r1;
            if (test1)
            {
                r1 = detail::erfc3_float(z);
                return test0 ? 2.f - r1 : r1;
            }
            z -= 0.4f;
            float r2 = exp(-x * x) * detail::erfc2_float(z);
            r1 = r2;
#ifndef XSIMD_NO_INFINITIES
            if (x == detail::inf_scalar(float()))
                r1 = 0.f;
#endif
            return test0 ? 2.f - r1 : r1;
        }

        XSIMD_INLINE double erfc(double self) noexcept
        {
            double x = detail::abs_scalar(self);
            double xx = x * x;
            const double lim1 = 0.65, lim2 = 2.2;
            bool test0 = self < 0.;
            bool test1 = x < lim1;
            double r1 = 0.;
            if (test1)
            {
                r1 = 1. - x * detail::erf1_double(xx);
                return test0 ? 2. - r1 : r1;
            }
            bool test2 = x < lim2;
            double ex = exp(-xx);
            if (test2)
            {
                double z = ex * detail::erfc2_double(x);
                return test0 ? 2. - z : z;
            }
            double z = ex * detail::erfc3_double(x);
            r1 = z;
#ifndef XSIMD_NO_INFINITIES
            if (x == detail::inf_scalar(double()))
                r1 = 0.;
#endif
            return test0 ? 2. - r1 : r1;
        }

        XSIMD_INLINE float tgamma(float self) noexcept
        {
            bool nan_result = (self < 0.f && detail::is_flint(self));
#ifndef XSIMD_NO_NANS
            nan_result = (self != self) || nan_result;
#endif
            float q = detail::abs_scalar(self);
            bool test = self < -33.f;
            if (test)
            {
                float r = detail::tgamma_large_negative(q);
                return nan_result ? detail::nan_like_batch<float>() : r;
            }
            float r2 = detail::tgamma_other(self);
#ifdef __FAST_MATH__
            return r2;
#else
            if (self == 0.f)
                return self < 0.f ? -detail::inf_scalar(float()) : detail::inf_scalar(float());
            return nan_result ? detail::nan_like_batch<float>() : r2;
#endif
        }

        XSIMD_INLINE double tgamma(double self) noexcept
        {
            bool nan_result = (self < 0. && detail::is_flint(self));
#ifndef XSIMD_NO_NANS
            nan_result = (self != self) || nan_result;
#endif
            double q = detail::abs_scalar(self);
            bool test = self < -33.;
            if (test)
            {
                double r = detail::tgamma_large_negative(q);
                return nan_result ? detail::nan_like_batch<double>() : r;
            }
            double r2 = detail::tgamma_other(self);
#ifdef __FAST_MATH__
            return r2;
#else
            if (self == 0.)
                return self < 0. ? -detail::inf_scalar(double()) : detail::inf_scalar(double());
            return nan_result ? detail::nan_like_batch<double>() : r2;
#endif
        }

        XSIMD_INLINE float lgamma(float a) noexcept
        {
            bool inf_result = (a <= 0.f) && detail::is_flint(a);
            float x = inf_result ? detail::nan_like_batch<float>() : a;
            float q = detail::abs_scalar(x);
#ifndef XSIMD_NO_INFINITIES
            inf_result = (x == detail::inf_scalar(float())) || inf_result;
#endif
            bool ltza = a < 0.f;
            if (ltza)
            {
#ifdef __FAST_MATH__
                float r = detail::lgamma_negative(q, detail::lgamma_other_float(q));
                return r;
#else
                float r = inf_result ? detail::inf_scalar(float()) : detail::lgamma_negative(q, detail::lgamma_other_float(q));
                if (a == -detail::inf_scalar(float()))
                    return detail::nan_like_batch<float>();
                return r;
#endif
            }
            float r2 = detail::lgamma_other_float(q);
#ifdef __FAST_MATH__
            return r2;
#else
            if (a == -detail::inf_scalar(float()))
                return detail::nan_like_batch<float>();
            return inf_result ? detail::inf_scalar(float()) : r2;
#endif
        }

        XSIMD_INLINE double lgamma(double a) noexcept
        {
            bool inf_result = (a <= 0.) && detail::is_flint(a);
            float _ = 0.f; (void)_;
            double x = inf_result ? detail::nan_like_batch<double>() : a;
            double q = detail::abs_scalar(x);
#ifndef XSIMD_NO_INFINITIES
            inf_result = (q == detail::inf_scalar(double()));
#endif
            bool test = a < -34.;
            if (test)
            {
                // large negative: log(pi) - log|q*sin(pi*q)| - lgamma(q)
                double st_q = detail::lgamma_other_double(q); // lgamma(|q|) via other()
                double p = detail::floor_scalar(q);
                double z = q - p;
                if (z < 0.5)
                    z = z - 1.;
                z = q * detail::sin_pi(z);
                z = detail::abs_scalar(z);
                double r = xsimd::constants::logpi<double>() - log(z) - st_q;
                return inf_result ? detail::nan_like_batch<double>() : r;
            }
            double r2 = detail::lgamma_other_double(a);
#ifdef __FAST_MATH__
            return r2;
#else
            if (a == -detail::inf_scalar(double()))
                return detail::nan_like_batch<double>();
            return inf_result ? detail::inf_scalar(double()) : r2;
#endif
        }

        /***********************************************************
         * Complex kernels: overloads of the core functions for    *
         * std::complex<float/double>, mirroring the batch complex  *
         * kernels and built on the real scalar kernels above.      *
         ***********************************************************/

        // forward declarations: the detail impls below are qualified calls
        template <class T>
        XSIMD_INLINE std::complex<T> log(const std::complex<T>& v) noexcept;
        template <class T>
        XSIMD_INLINE std::complex<T> sqrt(const std::complex<T>& v) noexcept;

        namespace detail
        {
            // Complex multiplication / division with the exact batch operator
            // semantics (fms/fma based multiply, naive component-wise divide),
            // so complex kernels are bit-identical to their batch counterparts.
            template <class T>
            XSIMD_INLINE std::complex<T> cmul(const std::complex<T>& a, const std::complex<T>& b) noexcept
            {
                T new_real = fms_scalar(a.real(), b.real(), a.imag() * b.imag());
                T new_imag = fma_scalar(a.real(), b.imag(), a.imag() * b.real());
                return std::complex<T>(new_real, new_imag);
            }

            template <class T>
            XSIMD_INLINE std::complex<T> cdiv(const std::complex<T>& a, const std::complex<T>& b) noexcept
            {
                T aa = a.real(), bb = a.imag(), c = b.real(), d = b.imag();
                // exact mirror of the complex log1p batch kernel division shape
                T e = fma_scalar(d, d, c * c);
                return std::complex<T>(fma_scalar(d, bb, c * aa) / e,
                                       fms_scalar(c, bb, d * aa) / e);
            }

            // complex sqrt mirroring the batch common kernel (rescaling variant)
            template <class T>
            XSIMD_INLINE std::complex<T> sqrt_complex_impl(const std::complex<T>& z) noexcept
            {
                constexpr T csqrt_scale_factor = std::is_same<T, float>::value ? T(6.7108864e7f) : T(1.8014398509481984e16);
                constexpr T csqrt_scale = std::is_same<T, float>::value ? T(1.220703125e-4f) : T(7.450580596923828125e-9);
                T x = z.real();
                T y = z.imag();
                T sqrt_x = sqrt_scalar(abs_scalar(x));
                T sqrt_hy = sqrt_scalar(T(0.5) * abs_scalar(y));
                bool cond = (abs_scalar(x) > T(4.) || abs_scalar(y) > T(4.));
                x = cond ? x * T(0.25) : x * csqrt_scale_factor;
                y = cond ? y * T(0.25) : y * csqrt_scale_factor;
                T scale = cond ? T(2.) : csqrt_scale;
                T r = hypot_scalar(x, y);

                bool condxp = x > T(0.);
                T t0 = condxp ? sqrt_scalar(T(0.5) * (r + x))
                              : sqrt_scalar(T(0.5) * (r - x));
                T r0 = scale * abs_scalar((T(0.5) * y) / t0);
                t0 *= scale;
                T t = condxp ? t0 : r0;
                r = condxp ? r0 : t0;
                std::complex<T> resg = y < T(0.) ? std::complex<T>(t, -r) : std::complex<T>(t, r);
                T ze(0.);

                if (y == ze)
                {
                    if (x == ze)
                        return std::complex<T>(ze, ze);
                    return x < ze ? std::complex<T>(ze, sqrt_x) : std::complex<T>(sqrt_x, ze);
                }
                if (x == ze)
                    return y > ze ? std::complex<T>(sqrt_hy, sqrt_hy) : std::complex<T>(sqrt_hy, -sqrt_hy);
                return resg;
            }

            // complex asin mirroring the batch common kernel
            template <class T>
            XSIMD_INLINE std::complex<T> asin_complex_impl(const std::complex<T>& z) noexcept
            {
                T x = z.real();
                T y = z.imag();

                std::complex<T> ct(-y, x);
                std::complex<T> zz(T(1.) - (x - y) * (x + y), -2 * x * y);
                zz = xsimd::scalar::log(ct + sqrt_complex_impl(zz));
                std::complex<T> resg(zz.imag(), -zz.real());

                if (y == T(0.))
                {
                    if (abs_scalar(x) > T(1.))
                        return std::complex<T>(xsimd::constants::pio2<T>(), T(0.));
                    return std::complex<T>(xsimd::scalar::asin(x), T(0.));
                }
                return resg;
            }

            // complex atan mirroring the batch common kernel
            template <class T>
            XSIMD_INLINE std::complex<T> atan_complex_impl(const std::complex<T>& z) noexcept
            {
                T x = z.real();
                T y = z.imag();
                T two_x = T(2.) * x;
                scalar_reassociation_barrier(two_x);
                T x2 = x * x;
                scalar_reassociation_barrier(x2);
                T one(1.);
                T y2 = y * y;
                scalar_reassociation_barrier(y2);
                T a = one - x2 - y2;
                scalar_reassociation_barrier(a);
                T atan2_res = xsimd::scalar::atan2(two_x, a);
                scalar_reassociation_barrier(atan2_res);
                T w = T(0.5) * atan2_res;
                scalar_reassociation_barrier(w);
                T num = y + one;
                num = fma_scalar(num, num, x2);
                T den = y - one;
                den = fma_scalar(den, den, x2);
                if ((x == T(0.)) && (y == T(1.)))
                    return std::complex<T>(T(0.), inf_scalar(T()));
                // batch computes 0.25 * log(num / den) on the real batch lane;
                // the products contract into fma exactly like the vectorized
                // kernel, so no barrier must break that here
                T nd = num / den;
                T lnd = xsimd::scalar::log(nd);
                scalar_reassociation_barrier(lnd);
                T imag = T(0.25) * lnd;
                scalar_reassociation_barrier(imag);
                return std::complex<T>(w, imag);
            }

            // Mirror of the batch pow kernel: z = exp(y*log|x|) with the
            // sign/odd-integer and zero handling reconstructed.
            template <class T>
            XSIMD_INLINE T pow_scalar_impl(T self, T other) noexcept
            {
                bool negself = self < T(0.);
                bool iszeropowpos = self == T(0.) && other >= T(0.);
                T adj_self = iszeropowpos ? T(1.) : abs_scalar(self);
                T z = xsimd::scalar::exp(other * xsimd::scalar::log(adj_self));
                z = iszeropowpos ? T(0.) : z;
                if (negself && is_odd_scalar(other))
                    z = -z;
                bool invalid = negself && !(is_flint(other) || other == inf_scalar(T()) || other == -inf_scalar(T()));
                return invalid ? nan_like_batch<T>() : z;
            }

            // Mirror of the batch complex pow kernel:
            // r = |a|^x, theta = x*arg(a); a^z = r*(cos theta, sin theta)
            template <class T>
            XSIMD_INLINE std::complex<T> pow_complex_impl(const std::complex<T>& a, const std::complex<T>& z) noexcept
            {
                T absa = hypot_scalar(a.real(), a.imag());
                T arga = xsimd::scalar::atan2(a.imag(), a.real());
                T x = z.real();
                T y = z.imag();
                T r = pow_scalar_impl(absa, x);
                T theta = x * arga;
                if (y != T(0.))
                {
                    r = r * xsimd::scalar::exp(-y * arga);
                    theta = theta + y * xsimd::scalar::log(absa);
                }
                T st = xsimd::scalar::sin(theta);
                T ct = xsimd::scalar::cos(theta);
                return absa == T(0.) ? std::complex<T>(T(0.), T(0.))
                                     : std::complex<T>(r * ct, r * st);
            }
        }

        // ---- complex trigonometric / hyperbolic ----

        template <class T>
        XSIMD_INLINE std::complex<T> sin(const std::complex<T>& z) noexcept
        {
            return std::complex<T>(xsimd::scalar::sin(z.real()) * xsimd::scalar::cosh(z.imag()),
                                   xsimd::scalar::cos(z.real()) * xsimd::scalar::sinh(z.imag()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> cos(const std::complex<T>& z) noexcept
        {
            return std::complex<T>(xsimd::scalar::cos(z.real()) * xsimd::scalar::cosh(z.imag()),
                                   -xsimd::scalar::sin(z.real()) * xsimd::scalar::sinh(z.imag()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> tan(const std::complex<T>& z) noexcept
        {
            T d = xsimd::scalar::cos(T(2) * z.real()) + xsimd::scalar::cosh(T(2) * z.imag());
            if (d == T(0.))
            {
                return std::complex<T>(detail::inf_scalar(T()), detail::inf_scalar(T()));
            }
            T wimag = xsimd::scalar::sinh(T(2) * z.imag());
            T wreal = xsimd::scalar::sin(T(2) * z.real()) / d;
            // when sinh overflows the imaginary part must saturate to +-1
            // rather than divide the infinity by d
            return std::complex<T>(wreal, wimag == detail::inf_scalar(T()) ? T(1) : wimag / d);
        }

        template <class T>
        XSIMD_INLINE std::complex<T> sinh(const std::complex<T>& z) noexcept
        {
            return std::complex<T>(xsimd::scalar::sinh(z.real()) * xsimd::scalar::cos(z.imag()),
                                   xsimd::scalar::cosh(z.real()) * xsimd::scalar::sin(z.imag()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> cosh(const std::complex<T>& z) noexcept
        {
            return std::complex<T>(xsimd::scalar::cosh(z.real()) * xsimd::scalar::cos(z.imag()),
                                   xsimd::scalar::sinh(z.real()) * xsimd::scalar::sin(z.imag()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> tanh(const std::complex<T>& z) noexcept
        {
            T d = xsimd::scalar::cosh(T(2) * z.real()) + xsimd::scalar::cos(T(2) * z.imag());
            return std::complex<T>(xsimd::scalar::sinh(T(2) * z.real()) / d,
                                   xsimd::scalar::sin(T(2) * z.imag()) / d);
        }

        // ---- complex exponential / logarithm ----

        template <class T>
        XSIMD_INLINE std::complex<T> exp(const std::complex<T>& z) noexcept
        {
            T er = xsimd::scalar::exp(z.real());
            return std::complex<T>(er * xsimd::scalar::cos(z.imag()), er * xsimd::scalar::sin(z.imag()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> expm1(const std::complex<T>& val) noexcept
        {
            T isin = xsimd::scalar::sin(val.imag());
            T rem1 = xsimd::scalar::expm1(val.real());
            T re = rem1 + T(1.);
            T si = xsimd::scalar::sin(val.imag() * T(0.5));
            return std::complex<T>(rem1 - T(2.) * re * si * si, re * isin);
        }

        // log(z) = log|z| + i*arg(z)
        template <class T>
        XSIMD_INLINE std::complex<T> log(const std::complex<T>& v) noexcept
        {
            T mod = detail::sqrt_scalar(detail::fma_scalar(v.real(), v.real(), v.imag() * v.imag()));
            return std::complex<T>(xsimd::scalar::log(mod), xsimd::scalar::atan2(v.imag(), v.real()));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> log1p(const std::complex<T>& val) noexcept
        {
            std::complex<T> u = std::complex<T>(T(1.), T(0.)) + val;
            if (u == std::complex<T>(T(1.), T(0.)))
                return val;
            std::complex<T> logu = xsimd::scalar::log(u);
            if (u.real() <= T(0.))
                return logu;
            std::complex<T> num = detail::cmul(logu, val);
            T num_re = num.real(), num_im = num.imag();
            detail::scalar_reassociation_barrier(num_re);
            detail::scalar_reassociation_barrier(num_im);
            std::complex<T> den = u - std::complex<T>(T(1.), T(0.));
            T den_re = den.real(), den_im = den.imag();
            detail::scalar_reassociation_barrier(den_re);
            detail::scalar_reassociation_barrier(den_im);
            return detail::cdiv(std::complex<T>(num_re, num_im), std::complex<T>(den_re, den_im));
        }

        // the batch kernel divides by the base packed as a complex value,
        // going through the complex division operator; mirror it.
        template <class T>
        XSIMD_INLINE std::complex<T> log2(const std::complex<T>& val) noexcept
        {
            return detail::cdiv(xsimd::scalar::log(val), std::complex<T>(T(0.693147180559945309417232121458176568), T(0.)));
        }

        template <class T>
        XSIMD_INLINE std::complex<T> log10(const std::complex<T>& val) noexcept
        {
            return detail::cdiv(xsimd::scalar::log(val), std::complex<T>(T(2.30258509299404568401799145468436421), T(0.)));
        }

        // ---- complex inverse trigonometric / hyperbolic ----

        template <class T>
        XSIMD_INLINE std::complex<T> asin(const std::complex<T>& v) noexcept
        {
            return detail::asin_complex_impl(v);
        }

        template <class T>
        XSIMD_INLINE std::complex<T> acos(const std::complex<T>& v) noexcept
        {
            std::complex<T> tmp = detail::asin_complex_impl(v);
            return std::complex<T>(xsimd::constants::pio2<T>() - tmp.real(), -tmp.imag());
        }

        template <class T>
        XSIMD_INLINE std::complex<T> atan(const std::complex<T>& v) noexcept
        {
            return detail::atan_complex_impl(v);
        }

        template <class T>
        XSIMD_INLINE std::complex<T> asinh(const std::complex<T>& v) noexcept
        {
            std::complex<T> w = detail::asin_complex_impl(std::complex<T>(-v.imag(), v.real()));
            return std::complex<T>(w.imag(), -w.real());
        }

        template <class T>
        XSIMD_INLINE std::complex<T> acosh(const std::complex<T>& v) noexcept
        {
            std::complex<T> w = detail::asin_complex_impl(v);
            std::complex<T> ac(xsimd::constants::pio2<T>() - w.real(), -w.imag());
            return std::complex<T>(-ac.imag(), ac.real());
        }

        template <class T>
        XSIMD_INLINE std::complex<T> atanh(const std::complex<T>& v) noexcept
        {
            std::complex<T> w = detail::atan_complex_impl(std::complex<T>(-v.imag(), v.real()));
            return std::complex<T>(w.imag(), -w.real());
        }

        // ---- complex sqrt / pow ----

        template <class T>
        XSIMD_INLINE std::complex<T> sqrt(const std::complex<T>& v) noexcept
        {
            return detail::sqrt_complex_impl(v);
        }

        template <class T>
        XSIMD_INLINE std::complex<T> pow(const std::complex<T>& a, const std::complex<T>& z) noexcept
        {
            return detail::pow_complex_impl(a, z);
        }

        // ---- real pow / sqrt public entry points ----

        template <class T>
        XSIMD_INLINE T pow(T self, T other) noexcept
        {
            return detail::pow_scalar_impl(self, other);
        }

        template <class T>
        XSIMD_INLINE T sqrt(T v) noexcept
        {
            return detail::sqrt_scalar(v);
        }
    }
}

#endif
