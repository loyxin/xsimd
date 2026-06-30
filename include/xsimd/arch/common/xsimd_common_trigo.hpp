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

#ifndef XSIMD_COMMON_TRIGO_HPP
#define XSIMD_COMMON_TRIGO_HPP

#include "./xsimd_common_details.hpp"

#include <array>
#include <complex>

namespace xsimd
{

    namespace kernel
    {
        /* origin: boost/simd/arch/common/detail/simd/trig_base.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        using namespace types;

        // acos
        template <class A, class T>
        XSIMD_INLINE batch<T, A> acos(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type x = abs(self);
            auto x_larger_05 = x > batch_type(0.5);
            x = select(x_larger_05, sqrt(fma(batch_type(-0.5), x, batch_type(0.5))), self);
            x = asin(x);
            x = select(x_larger_05, x + x, x);
            x = select(self < batch_type(-0.5), constants::pi<batch_type>() - x, x);
            return select(x_larger_05, x, constants::pio2<batch_type>() - x);
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> acos(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            using real_batch = typename batch_type::real_batch;
            batch_type tmp = asin(z);
            return { constants::pio2<real_batch>() - tmp.real(), -tmp.imag() };
        }

        // acosh
        /* origin: boost/simd/arch/common/simd/function/acosh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class A, class T>
        XSIMD_INLINE batch<T, A> acosh(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type x = self - batch_type(1.);
            auto test = x > constants::oneotwoeps<batch_type>();
            batch_type z = select(test, self, x + sqrt(x + x + x * x));
            batch_type l1pz = log1p(z);
            return select(test, l1pz + constants::log_2<batch_type>(), l1pz);
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> acosh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            batch_type w = acos(z);
            w = batch_type(-w.imag(), w.real());
            return w;
        }

        // asin
        template <class A>
        XSIMD_INLINE batch<float, A> asin(batch<float, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<float, A>;
            batch_type x = abs(self);
            batch_type sign = bitofsign(self);
            auto x_larger_05 = x > batch_type(0.5);
            batch_type z = select(x_larger_05, batch_type(0.5) * (batch_type(1.) - x), x * x);
            x = select(x_larger_05, sqrt(z), x);
            batch_type z1 = detail::horner<batch_type,
                                           0x3e2aaae4,
                                           0x3d9980f6,
                                           0x3d3a3ec7,
                                           0x3cc617e3,
                                           0x3d2cb352>(z);
            z1 = fma(z1, z * x, x);
            z = select(x_larger_05, constants::pio2<batch_type>() - (z1 + z1), z1);
            return z ^ sign;
        }
        template <class A>
        XSIMD_INLINE batch<double, A> asin(batch<double, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<double, A>;
            batch_type x = abs(self);
            auto small_cond = x < constants::sqrteps<batch_type>();
            batch_type ct1 = batch_type(bit_cast<double>(int64_t(0x3fe4000000000000)));
            batch_type zz1 = batch_type(1.) - x;
            batch_type vp = zz1 * detail::horner<batch_type, 0x403c896240f3081dull, 0xc03991aaac01ab68ull, 0x401bdff5baf33e6aull, 0xbfe2079259f9290full, 0x3f684fc3988e9f08ull>(zz1) / detail::horner1<batch_type, 0x40756709b0b644beull, 0xc077fe08959063eeull, 0x40626219af6a7f42ull, 0xc035f2a2b6bf5d8cull>(zz1);
            zz1 = sqrt(zz1 + zz1);
            batch_type z = constants::pio4<batch_type>() - zz1;
            zz1 = fms(zz1, vp, constants::pio_2lo<batch_type>());
            z = z - zz1;
            zz1 = z + constants::pio4<batch_type>();
            batch_type zz2 = self * self;
            z = zz2 * detail::horner<batch_type, 0xc020656c06ceafd5ull, 0x40339007da779259ull, 0xc0304331de27907bull, 0x4015c74b178a2dd9ull, 0xbfe34341333e5c16ull, 0x3f716b9b0bd48ad3ull>(zz2) / detail::horner1<batch_type, 0xc04898220a3607acull, 0x4061705684ffbf9dull, 0xc06265bb6d3576d7ull, 0x40519fc025fe9054ull, 0xc02d7b590b5e0eabull>(zz2);
            zz2 = fma(x, z, x);
            return select(x > batch_type(1.), constants::nan<batch_type>(),
                          select(small_cond, x,
                                 select(x > ct1, zz1, zz2))
                              ^ bitofsign(self));
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> asin(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            using real_batch = typename batch_type::real_batch;
            real_batch x = z.real();
            real_batch y = z.imag();

            batch_type ct(-y, x);
            batch_type zz(real_batch(1.) - (x - y) * (x + y), -2 * x * y);
            zz = log(ct + sqrt(zz));
            batch_type resg(zz.imag(), -zz.real());

            return select(y == real_batch(0.),
                          select(fabs(x) > real_batch(1.),
                                 batch_type(constants::pio2<real_batch>(), real_batch(0.)),
                                 batch_type(asin(x), real_batch(0.))),
                          resg);
        }

        // asinh
        /* origin: boost/simd/arch/common/simd/function/asinh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        namespace detail
        {
            template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
            XSIMD_INLINE batch<T, A>
            average(const batch<T, A>& x1, const batch<T, A>& x2) noexcept
            {
                return (x1 & x2) + ((x1 ^ x2) >> 1);
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A>
            averagef(const batch<T, A>& x1, const batch<T, A>& x2) noexcept
            {
                using batch_type = batch<T, A>;
                return fma(x1, batch_type(0.5), x2 * batch_type(0.5));
            }
            template <class A>
            XSIMD_INLINE batch<float, A> average(batch<float, A> const& x1, batch<float, A> const& x2) noexcept
            {
                return averagef(x1, x2);
            }
            template <class A>
            XSIMD_INLINE batch<double, A> average(batch<double, A> const& x1, batch<double, A> const& x2) noexcept
            {
                return averagef(x1, x2);
            }
        }
        template <class A>
        XSIMD_INLINE batch<float, A> asinh(batch<float, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<float, A>;
            batch_type x = abs(self);
            auto lthalf = x < batch_type(0.5);
            batch_type x2 = x * x;
            batch_type bts = bitofsign(self);
            batch_type z(0.);
            if (any(lthalf))
            {
                z = detail::horner<batch_type,
                                   0x3f800000,
                                   0xbe2aa9ad,
                                   0x3d9949b1,
                                   0xbd2ee581,
                                   0x3ca4d6e6>(x2)
                    * x;
                if (all(lthalf))
                    return z ^ bts;
            }
            batch_type tmp = select(x > constants::oneosqrteps<batch_type>(), x, detail::average(x, hypot(batch_type(1.), x)));
#ifndef XSIMD_NO_NANS
            return select(isnan(self), constants::nan<batch_type>(), select(lthalf, z, log(tmp) + constants::log_2<batch_type>()) ^ bts);
#else
            return select(lthalf, z, log(tmp) + constants::log_2<batch_type>()) ^ bts;
#endif
        }
        template <class A>
        XSIMD_INLINE batch<double, A> asinh(batch<double, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<double, A>;
            batch_type x = abs(self);
            auto test = x > constants::oneosqrteps<batch_type>();
            batch_type z = select(test, x - batch_type(1.), x + x * x / (batch_type(1.) + hypot(batch_type(1.), x)));
#ifndef XSIMD_NO_INFINITIES
            z = select(x == constants::infinity<batch_type>(), x, z);
#endif
            batch_type l1pz = log1p(z);
            z = select(test, l1pz + constants::log_2<batch_type>(), l1pz);
            return bitofsign(self) ^ z;
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> asinh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            batch_type w = asin(batch_type(-z.imag(), z.real()));
            w = batch_type(w.imag(), -w.real());
            return w;
        }

        // atan
        namespace detail
        {
            template <class A>
            static XSIMD_INLINE batch<float, A> kernel_atan(const batch<float, A>& x, const batch<float, A>& recx) noexcept
            {
                using batch_type = batch<float, A>;
                const auto flag1 = x < constants::tan3pio8<batch_type>();
                const auto flag2 = (x >= batch_type(bit_cast<float>((uint32_t)0x3ed413cd))) && flag1;
                batch_type yy = select(flag1, batch_type(0.), constants::pio2<batch_type>());
                yy = select(flag2, constants::pio4<batch_type>(), yy);
                batch_type xx = select(flag1, x, -recx);
                xx = select(flag2, (x - batch_type(1.)) / (x + batch_type(1.)), xx);
                const batch_type z = xx * xx;
                batch_type z1 = detail::horner<batch_type,
                                               0xbeaaaa2aul,
                                               0x3e4c925ful,
                                               0xbe0e1b85ul,
                                               0x3da4f0d1ul>(z);
                z1 = fma(xx, z1 * z, xx);
                z1 = select(flag2, z1 + constants::pio_4lo<batch_type>(), z1);
                z1 = select(!flag1, z1 + constants::pio_2lo<batch_type>(), z1);
                return yy + z1;
            }
            template <class A>
            static XSIMD_INLINE batch<double, A> kernel_atan(const batch<double, A>& x, const batch<double, A>& recx) noexcept
            {
                using batch_type = batch<double, A>;
                const auto flag1 = x < constants::tan3pio8<batch_type>();
                const auto flag2 = (x >= constants::tanpio8<batch_type>()) && flag1;
                batch_type yy = select(flag1, batch_type(0.), constants::pio2<batch_type>());
                yy = select(flag2, constants::pio4<batch_type>(), yy);
                batch_type xx = select(flag1, x, -recx);
                xx = select(flag2, (x - batch_type(1.)) / (x + batch_type(1.)), xx);
                batch_type z = xx * xx;
                z *= detail::horner<batch_type,
                                    0xc0503669fd28ec8eull,
                                    0xc05eb8bf2d05ba25ull,
                                    0xc052c08c36880273ull,
                                    0xc03028545b6b807aull,
                                    0xbfec007fa1f72594ull>(z)
                    / detail::horner1<batch_type,
                                      0x4068519efbbd62ecull,
                                      0x407e563f13b049eaull,
                                      0x407b0e18d2e2be3bull,
                                      0x4064a0dd43b8fa25ull,
                                      0x4038dbc45b14603cull>(z);
                z = fma(xx, z, xx);
                z = select(flag2, z + constants::pio_4lo<batch_type>(), z);
                z = z + select(flag1, batch_type(0.), constants::pio_2lo<batch_type>());
                return yy + z;
            }
        }
        template <class A, class T>
        XSIMD_INLINE batch<T, A> atan(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            const batch_type absa = abs(self);
            const batch_type x = detail::kernel_atan(absa, batch_type(1.) / absa);
            return x ^ bitofsign(self);
        }

        // acot
        template <class A, class T>
        XSIMD_INLINE batch<T, A> acot(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            return constants::pio2<batch_type>() - atan(self);
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> atan(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            using real_batch = typename batch_type::real_batch;
            real_batch x = z.real();
            real_batch y = z.imag();
            real_batch x2 = x * x;
            real_batch one(1.);
            real_batch a = one - x2 - (y * y);
            real_batch w = 0.5 * atan2(2. * x, a);
            real_batch num = y + one;
            num = x2 + num * num;
            real_batch den = y - one;
            den = x2 + den * den;
#ifdef __FAST_MATH__
            return batch_type(w, 0.25 * log(num / den));
#else
            return select((x == real_batch(0.)) && (y == real_batch(1.)),
                          batch_type(real_batch(0.), constants::infinity<real_batch>()),
                          batch_type(w, 0.25 * log(num / den)));
#endif
        }

        // atanh
        /* origin: boost/simd/arch/common/simd/function/acosh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class A, class T>
        XSIMD_INLINE batch<T, A> atanh(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type x = abs(self);
            batch_type t = x + x;
            batch_type z = batch_type(1.) - x;
            auto test = x < batch_type(0.5);
            batch_type tmp = select(test, x, t) / z;
            return bitofsign(self) ^ (batch_type(0.5) * log1p(select(test, fma(t, tmp, t), tmp)));
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> atanh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            batch_type w = atan(batch_type(-z.imag(), z.real()));
            w = batch_type(w.imag(), -w.real());
            return w;
        }

        // atan2
        template <class A, class T>
        XSIMD_INLINE batch<T, A> atan2(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
#ifdef __FAST_MATH__
            const batch_type q = abs(self / other);
            const batch_type q_p = abs(other / self);
#else
            const batch_type q = abs(self / other);
            const batch_type q_p = 1. / q;
#endif
            const batch_type z = detail::kernel_atan(q, q_p);
            return select(other > batch_type(0.), z, constants::pi<batch_type>() - z) * signnz(self);
        }

        // cos
        namespace detail
        {
            template <class T, class A>
            XSIMD_INLINE batch<T, A> quadrant(const batch<T, A>& x) noexcept
            {
                return x & batch<T, A>(3);
            }

            template <class A>
            XSIMD_INLINE batch<float, A> quadrant(const batch<float, A>& x) noexcept
            {
                return to_float(quadrant(to_int(x)));
            }

            template <class A>
            XSIMD_INLINE batch<double, A> quadrant(const batch<double, A>& x) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type a = x * batch_type(0.25);
                return (a - floor(a)) * batch_type(4.);
            }
            /* origin: boost/simd/arch/common/detail/simd/f_trig_evaluation.hpp */
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */

            template <class A>
            XSIMD_INLINE batch<float, A> cos_eval(const batch<float, A>& z) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type y = detail::horner<batch_type,
                                              0x3d2aaaa5,
                                              0xbab60619,
                                              0x37ccf5ce>(z);
                return batch_type(1.) + fma(z, batch_type(-0.5), y * z * z);
            }

            template <class A>
            XSIMD_INLINE batch<float, A> sin_eval(const batch<float, A>& z, const batch<float, A>& x) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type y = detail::horner<batch_type,
                                              0xbe2aaaa2,
                                              0x3c08839d,
                                              0xb94ca1f9>(z);
                return fma(y * z, x, x);
            }

            template <class A>
            static XSIMD_INLINE batch<float, A> base_tancot_eval(const batch<float, A>& z) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type zz = z * z;
                batch_type y = detail::horner<batch_type,
                                              0x3eaaaa6f,
                                              0x3e0896dd,
                                              0x3d5ac5c9,
                                              0x3cc821b5,
                                              0x3b4c779c,
                                              0x3c19c53b>(zz);
                return fma(y, zz * z, z);
            }

            template <class A, class BB>
            static XSIMD_INLINE batch<float, A> tan_eval(const batch<float, A>& z, const BB& test) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, y, -batch_type(1.) / y);
            }

            template <class A, class BB>
            static XSIMD_INLINE batch<float, A> cot_eval(const batch<float, A>& z, const BB& test) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, batch_type(1.) / y, -y);
            }

            /* origin: boost/simd/arch/common/detail/simd/d_trig_evaluation.hpp */
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */
            template <class A>
            static XSIMD_INLINE batch<double, A> cos_eval(const batch<double, A>& z) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type y = detail::horner<batch_type,
                                              0x3fe0000000000000ull,
                                              0xbfa5555555555551ull,
                                              0x3f56c16c16c15d47ull,
                                              0xbefa01a019ddbcd9ull,
                                              0x3e927e4f8e06d9a5ull,
                                              0xbe21eea7c1e514d4ull,
                                              0x3da8ff831ad9b219ull>(z);
                return batch_type(1.) - y * z;
            }

            template <class A>
            static XSIMD_INLINE batch<double, A> sin_eval(const batch<double, A>& z, const batch<double, A>& x) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type y = detail::horner<batch_type,
                                              0xbfc5555555555548ull,
                                              0x3f8111111110f7d0ull,
                                              0xbf2a01a019bfdf03ull,
                                              0x3ec71de3567d4896ull,
                                              0xbe5ae5e5a9291691ull,
                                              0x3de5d8fd1fcf0ec1ull>(z);
                return fma(y * z, x, x);
            }

            template <class A>
            static XSIMD_INLINE batch<double, A> base_tancot_eval(const batch<double, A>& z) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type zz = z * z;
                batch_type num = detail::horner<batch_type,
                                                0xc1711fead3299176ull,
                                                0x413199eca5fc9dddull,
                                                0xc0c992d8d24f3f38ull>(zz);
                batch_type den = detail::horner1<batch_type,
                                                 0xc189afe03cbe5a31ull,
                                                 0x4177d98fc2ead8efull,
                                                 0xc13427bc582abc96ull,
                                                 0x40cab8a5eeb36572ull>(zz);
                return fma(z, (zz * (num / den)), z);
            }

            template <class A, class BB>
            static XSIMD_INLINE batch<double, A> tan_eval(const batch<double, A>& z, const BB& test) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, y, -batch_type(1.) / y);
            }

            template <class A, class BB>
            static XSIMD_INLINE batch<double, A> cot_eval(const batch<double, A>& z, const BB& test) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type y = base_tancot_eval(z);
                return select(test, batch_type(1.) / y, -y);
            }
            /* origin: boost/simd/arch/common/detail/simd/trig_reduction.hpp */
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */

            struct trigo_radian_tag
            {
            };
            struct trigo_pi_tag
            {
            };

            template <class B, class Tag = trigo_radian_tag>
            struct trigo_reducer
            {
                static XSIMD_INLINE B reduce(const B& x, B& xr) noexcept
                {
                    if (all(x <= constants::pio4<B>()))
                    {
                        xr = x;
                        return B(0.);
                    }
                    else if (all(x <= constants::pio2<B>()))
                    {
                        auto test = x > constants::pio4<B>();
                        xr = x - constants::pio2_1<B>();
                        detail::reassociation_barrier(xr, "ordered pio2 subtraction");
                        xr -= constants::pio2_2<B>();
                        detail::reassociation_barrier(xr, "ordered pio2 subtraction");
                        xr -= constants::pio2_3<B>();
                        detail::reassociation_barrier(xr, "ordered pio2 subtraction");
                        xr = select(test, xr, x);
                        return select(test, B(1.), B(0.));
                    }
                    else if (all(x <= constants::twentypi<B>()))
                    {
                        B xi = nearbyint(x * constants::twoopi<B>());
                        detail::reassociation_barrier(xi, "preserve quadrant selection");
                        xr = fnma(xi, constants::pio2_1<B>(), x);
                        detail::reassociation_barrier(xr, "compensated range reduction");
                        xr -= xi * constants::pio2_2<B>();
                        detail::reassociation_barrier(xr, "compensated range reduction");
                        xr -= xi * constants::pio2_3<B>();
                        detail::reassociation_barrier(xr, "compensated range reduction");
                        return quadrant(xi);
                    }
                    else if (all(x <= constants::mediumpi<B>()))
                    {
                        B fn = nearbyint(x * constants::twoopi<B>());
                        detail::reassociation_barrier(fn, "multi-term range reduction");
                        B r = x - fn * constants::pio2_1<B>();
                        detail::reassociation_barrier(r, "multi-term range reduction");
                        B w = fn * constants::pio2_1t<B>();
                        B t = r;
                        w = fn * constants::pio2_2<B>();
                        r = t - w;
                        detail::reassociation_barrier(r, "multi-term range reduction");
                        w = fn * constants::pio2_2t<B>() - ((t - r) - w);
                        t = r;
                        w = fn * constants::pio2_3<B>();
                        r = t - w;
                        detail::reassociation_barrier(r, "multi-term range reduction");
                        w = fn * constants::pio2_3t<B>() - ((t - r) - w);
                        xr = r - w;
                        detail::reassociation_barrier(xr, "multi-term range reduction");
                        return quadrant(fn);
                    }
                    else
                    {
                        static constexpr std::size_t size = B::size;
                        using value_type = typename B::value_type;
                        alignas(B) std::array<value_type, size> tmp;
                        alignas(B) std::array<value_type, size> txr;
                        alignas(B) std::array<value_type, size> args;
                        x.store_aligned(args.data());

                        for (std::size_t i = 0; i < size; ++i)
                        {
                            double arg = args[i];
#ifndef __FAST_MATH__
                            if (arg == std::numeric_limits<value_type>::infinity())
                            {
                                tmp[i] = 0.;
                                txr[i] = std::numeric_limits<value_type>::quiet_NaN();
                            }
                            else
#endif
                            {
                                double y[2];
                                std::int32_t n = ::xsimd::detail::__ieee754_rem_pio2(arg, y);
                                tmp[i] = value_type(n & 3);
                                txr[i] = value_type(y[0]);
                            }
                        }
                        xr = B::load_aligned(&txr[0]);
                        B res = B::load_aligned(&tmp[0]);
                        return res;
                    }
                }
            };

            template <class B>
            struct trigo_reducer<B, trigo_pi_tag>
            {
                static XSIMD_INLINE B reduce(const B& x, B& xr) noexcept
                {
                    B xi = nearbyint(x * B(2.));
                    B x2 = x - xi * B(0.5);
                    xr = x2 * constants::pi<B>();
                    return quadrant(xi);
                }
            };

        }
        template <class A, class T>
        XSIMD_INLINE batch<T, A> cos(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            const batch_type x = abs(self);
            batch_type xr = constants::nan<batch_type>();
            const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
            auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
            auto swap_bit = fma(batch_type(-2.), tmp, n);
            auto sign_bit = select((swap_bit ^ tmp) != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
            const batch_type z = xr * xr;
            const batch_type se = detail::sin_eval(z, xr);
            const batch_type ce = detail::cos_eval(z);
            const batch_type z1 = select(swap_bit != batch_type(0.), se, ce);
            return z1 ^ sign_bit;
        }

        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> cos(batch<std::complex<T>, A> const& z, requires_arch<common>) noexcept
        {
            return { cos(z.real()) * cosh(z.imag()), -sin(z.real()) * sinh(z.imag()) };
        }

        // cosh

        /* origin: boost/simd/arch/common/simd/function/cosh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        template <class A, class T>
        XSIMD_INLINE batch<T, A> cosh(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type x = abs(self);
            auto test1 = x > (constants::maxlog<batch_type>() - constants::log_2<batch_type>());
            batch_type fac = select(test1, batch_type(0.5), batch_type(1.));
            batch_type tmp = exp(x * fac);
            batch_type tmp1 = batch_type(0.5) * tmp;
            return select(test1, tmp1 * tmp, detail::average(tmp, batch_type(1.) / tmp));
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> cosh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            auto x = z.real();
            auto y = z.imag();
            return { cosh(x) * cos(y), sinh(x) * sin(y) };
        }

        // sin
        namespace detail
        {
            template <class A, class T, class Tag = trigo_radian_tag>
            XSIMD_INLINE batch<T, A> sin(batch<T, A> const& self, Tag = Tag()) noexcept
            {
                using batch_type = batch<T, A>;
                const batch_type x = abs(self);
                batch_type xr = constants::nan<batch_type>();
                const batch_type n = detail::trigo_reducer<batch_type, Tag>::reduce(x, xr);
                auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
                auto swap_bit = fma(batch_type(-2.), tmp, n);
                auto sign_bit = bitofsign(self) ^ select(tmp != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
                const batch_type z = xr * xr;
                const batch_type se = detail::sin_eval(z, xr);
                const batch_type ce = detail::cos_eval(z);
                const batch_type z1 = select(swap_bit == batch_type(0.), se, ce);
                return z1 ^ sign_bit;
            }
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> sin(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            return detail::sin(self);
        }

        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> sin(batch<std::complex<T>, A> const& z, requires_arch<common>) noexcept
        {
            return { sin(z.real()) * cosh(z.imag()), cos(z.real()) * sinh(z.imag()) };
        }

        // sincos
        template <class A, class T>
        XSIMD_INLINE std::pair<batch<T, A>, batch<T, A>> sincos(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            const batch_type x = abs(self);
            batch_type xr = constants::nan<batch_type>();
            const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
            auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
            auto swap_bit = fma(batch_type(-2.), tmp, n);
            const batch_type z = xr * xr;
            const batch_type se = detail::sin_eval(z, xr);
            const batch_type ce = detail::cos_eval(z);
            auto sin_sign_bit = bitofsign(self) ^ select(tmp != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
            const batch_type sin_z1 = select(swap_bit == batch_type(0.), se, ce);
            auto cos_sign_bit = select((swap_bit ^ tmp) != batch_type(0.), constants::signmask<batch_type>(), batch_type(0.));
            const batch_type cos_z1 = select(swap_bit != batch_type(0.), se, ce);
            return std::make_pair(sin_z1 ^ sin_sign_bit, cos_z1 ^ cos_sign_bit);
        }

        template <class A, class T>
        XSIMD_INLINE std::pair<batch<std::complex<T>, A>, batch<std::complex<T>, A>>
        sincos(batch<std::complex<T>, A> const& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            using real_batch = typename batch_type::real_batch;
            real_batch rcos = cos(z.real());
            real_batch rsin = sin(z.real());
            real_batch icosh = cosh(z.imag());
            real_batch isinh = sinh(z.imag());
            return std::make_pair(batch_type(rsin * icosh, rcos * isinh), batch_type(rcos * icosh, -rsin * isinh));
        }

        // sinh
        namespace detail
        {
            /* origin: boost/simd/arch/common/detail/common/sinh_kernel.hpp */
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */
            template <class A>
            XSIMD_INLINE batch<float, A> sinh_kernel(batch<float, A> const& self) noexcept
            {
                using batch_type = batch<float, A>;
                batch_type sqr_self = self * self;
                return detail::horner<batch_type,
                                      0x3f800000, // 1.0f
                                      0x3e2aaacc, // 1.66667160211E-1f
                                      0x3c087bbe, // 8.33028376239E-3f
                                      0x39559e2f // 2.03721912945E-4f
                                      >(sqr_self)
                    * self;
            }

            template <class A>
            XSIMD_INLINE batch<double, A> sinh_kernel(batch<double, A> const& self) noexcept
            {
                using batch_type = batch<double, A>;
                batch_type sqrself = self * self;
                return fma(self, (detail::horner<batch_type,
                                                 0xc115782bdbf6ab05ull, //  -3.51754964808151394800E5
                                                 0xc0c694b8c71d6182ull, //  -1.15614435765005216044E4,
                                                 0xc064773a398ff4feull, //  -1.63725857525983828727E2,
                                                 0xbfe9435fe8bb3cd6ull //  -7.89474443963537015605E-1
                                                 >(sqrself)
                                  / detail::horner1<batch_type,
                                                    0xc1401a20e4f90044ull, //  -2.11052978884890840399E6
                                                    0x40e1a7ba7ed72245ull, //   3.61578279834431989373E4,
                                                    0xc0715b6096e96484ull //  -2.77711081420602794433E2,
                                                    >(sqrself))
                               * sqrself,
                           self);
            }
        }
        /* origin: boost/simd/arch/common/simd/function/sinh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class A, class T>
        XSIMD_INLINE batch<T, A> sinh(batch<T, A> const& a, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type half(0.5);
            batch_type x = abs(a);
            auto lt1 = x < batch_type(1.);
            batch_type bts = bitofsign(a);
            batch_type z(0.);
            if (any(lt1))
            {
                z = detail::sinh_kernel(x);
                if (all(lt1))
                    return z ^ bts;
            }
            auto test1 = x > (constants::maxlog<batch_type>() - constants::log_2<batch_type>());
            batch_type fac = select(test1, half, batch_type(1.));
            batch_type tmp = exp(x * fac);
            batch_type tmp1 = half * tmp;
            batch_type r = select(test1, tmp1 * tmp, tmp1 - half / tmp);
            return select(lt1, z, r) ^ bts;
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> sinh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            auto x = z.real();
            auto y = z.imag();
            return { sinh(x) * cos(y), cosh(x) * sin(y) };
        }

        // tan
        template <class A, class T>
        XSIMD_INLINE batch<T, A> tan(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            const batch_type x = abs(self);
            batch_type xr = constants::nan<batch_type>();
            const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
            auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
            auto swap_bit = fma(batch_type(-2.), tmp, n);
            auto test = (swap_bit == batch_type(0.));
            const batch_type y = detail::tan_eval(xr, test);
            return y ^ bitofsign(self);
        }

        // cot
        template <class A, class T>
        XSIMD_INLINE batch<T, A> cot(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            const batch_type x = abs(self);
            batch_type xr = constants::nan<batch_type>();
            const batch_type n = detail::trigo_reducer<batch_type>::reduce(x, xr);
            auto tmp = select(n >= batch_type(2.), batch_type(1.), batch_type(0.));
            auto swap_bit = fma(batch_type(-2.), tmp, n);
            auto test = (swap_bit == batch_type(0.));
            const batch_type y = detail::cot_eval(xr, test);
            return bitwise_xor(y, bitwise_and(self, constants::signmask<batch_type>()));
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> tan(batch<std::complex<T>, A> const& z, requires_arch<common>) noexcept
        {
            using batch_type = batch<std::complex<T>, A>;
            using real_batch = typename batch_type::real_batch;
            real_batch d = cos(2 * z.real()) + cosh(2 * z.imag());
            real_batch wreal = sin(2 * z.real()) / d;
            real_batch wimag = sinh(2 * z.imag());
#ifdef __FAST_MATH__
            return batch_type(wreal, real_batch(1.)), batch_type(wreal, wimag / d);
#else
            batch_type winf(constants::infinity<real_batch>(), constants::infinity<real_batch>());
            batch_type wres = select(isinf(wimag), batch_type(wreal, real_batch(1.)), batch_type(wreal, wimag / d));
            return select(d == real_batch(0.), winf, wres);
#endif
        }

        // tanh
        namespace detail
        {
            /* origin: boost/simd/arch/common/detail/common/tanh_kernel.hpp */
            /*
             * ====================================================
             * copyright 2016 NumScale SAS
             *
             * Distributed under the Boost Software License, Version 1.0.
             * (See copy at http://boost.org/LICENSE_1_0.txt)
             * ====================================================
             */
            template <class B>
            struct tanh_kernel;

            template <class A>
            struct tanh_kernel<batch<float, A>>
            {
                using batch_type = batch<float, A>;
                static XSIMD_INLINE batch_type tanh(const batch_type& x) noexcept
                {
                    batch_type sqrx = x * x;
                    return fma(detail::horner<batch_type,
                                              0xbeaaaa99, //    -3.33332819422E-1F
                                              0x3e088393, //    +1.33314422036E-1F
                                              0xbd5c1e2d, //    -5.37397155531E-2F
                                              0x3ca9134e, //    +2.06390887954E-2F
                                              0xbbbaf0ea //    -5.70498872745E-3F
                                              >(sqrx)
                                   * sqrx,
                               x, x);
                }

                static XSIMD_INLINE batch_type cotanh(const batch_type& x) noexcept
                {
                    return batch_type(1.) / tanh(x);
                }
            };

            template <class A>
            struct tanh_kernel<batch<double, A>>
            {
                using batch_type = batch<double, A>;
                static XSIMD_INLINE batch_type tanh(const batch_type& x) noexcept
                {
                    batch_type sqrx = x * x;
                    return fma(sqrx * p(sqrx) / q(sqrx), x, x);
                }

                static XSIMD_INLINE batch_type cotanh(const batch_type& x) noexcept
                {
                    batch_type sqrx = x * x;
                    batch_type qval = q(sqrx);
                    return qval / (x * fma(p(sqrx), sqrx, qval));
                }

                static XSIMD_INLINE batch_type p(const batch_type& x) noexcept
                {
                    return detail::horner<batch_type,
                                          0xc0993ac030580563, // -1.61468768441708447952E3
                                          0xc058d26a0e26682d, // -9.92877231001918586564E1,
                                          0xbfeedc5baafd6f4b // -9.64399179425052238628E-1
                                          >(x);
                }

                static XSIMD_INLINE batch_type q(const batch_type& x) noexcept
                {
                    return detail::horner1<batch_type,
                                           0x40b2ec102442040c, //  4.84406305325125486048E3
                                           0x40a176fa0e5535fa, //  2.23548839060100448583E3,
                                           0x405c33f28a581B86 //  1.12811678491632931402E2,
                                           >(x);
                }
            };

        }
        /* origin: boost/simd/arch/common/simd/function/tanh.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class A, class T>
        XSIMD_INLINE batch<T, A> tanh(batch<T, A> const& self, requires_arch<common>) noexcept
        {
            using batch_type = batch<T, A>;
            batch_type one(1.);
            batch_type x = abs(self);
            auto test = x < (batch_type(5.) / batch_type(8.));
            batch_type bts = bitofsign(self);
            batch_type z = one;
            if (any(test))
            {
                z = detail::tanh_kernel<batch_type>::tanh(x);
                if (all(test))
                    return z ^ bts;
            }
            batch_type r = fma(batch_type(-2.), one / (one + exp(x + x)), one);
            return select(test, z, r) ^ bts;
        }
        template <class A, class T>
        XSIMD_INLINE batch<std::complex<T>, A> tanh(const batch<std::complex<T>, A>& z, requires_arch<common>) noexcept
        {
            using real_batch = typename batch<std::complex<T>, A>::real_batch;
            auto x = z.real();
            auto y = z.imag();
            real_batch two(2);
            auto d = cosh(two * x) + cos(two * y);
            return { sinh(two * x) / d, sin(two * y) / d };
        }

    }

    namespace detail
    {
        template <class T>
        XSIMD_INLINE T horner_scalar(T) noexcept
        {
            return T(0.);
        }

        template <class T, uint64_t c0>
        XSIMD_INLINE T horner_scalar(T) noexcept
        {
            return ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(c0));
        }

        template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
        XSIMD_INLINE T horner_scalar(T x) noexcept
        {
            return x * horner_scalar<T, c1, args...>(x) + ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(c0));
        }

        XSIMD_INLINE float scalar_cos_eval(float z) noexcept
        {
            float y = horner_scalar<float,
                                     0x3d2aaaa5,
                                     0xbab60619,
                                     0x37ccf5ce>(z);
            return 1.f + (z * -0.5f + y * z * z);
        }

        XSIMD_INLINE float scalar_sin_eval(float z, float x) noexcept
        {
            float y = horner_scalar<float,
                                    0xbe2aaaa2,
                                    0x3c08839d,
                                    0xb94ca1f9>(z);
            return (y * z) * x + x;
        }

        XSIMD_INLINE double scalar_cos_eval(double z) noexcept
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

        XSIMD_INLINE double scalar_sin_eval(double z, double x) noexcept
        {
            double y = horner_scalar<double,
                                     0xbfc5555555555548ull,
                                     0x3f8111111110f7d0ull,
                                     0xbf2a01a019bfdf03ull,
                                     0x3ec71de3567d4896ull,
                                     0xbe5ae5e5a9291691ull,
                                     0x3de5d8fd1fcf0ec1ull>(z);
            return (y * z) * x + x;
        }

        template <class T>
        XSIMD_INLINE T trigo_reduce_scalar(T x, int& quadrant) noexcept;

        template <>
        XSIMD_INLINE float trigo_reduce_scalar<float>(float x, int& quadrant) noexcept
        {
            namespace tc = xsimd::constants;

            if (x <= tc::pio4<float>())
            {
                quadrant = 0;
                return x;
            }
            else if (x <= tc::pio2<float>())
            {
                quadrant = (x > tc::pio4<float>()) ? 1 : 0;
                float xr = x - tc::pio2_1<float>();
                xr -= tc::pio2_2<float>();
                xr -= tc::pio2_3<float>();
                return quadrant ? xr : x;
            }
            else if (x <= tc::twentypi<float>())
            {
                float xi = std::nearbyint(x * tc::twoopi<float>());
                float xr = -xi * tc::pio2_1<float>() + x;
                xr -= xi * tc::pio2_2<float>();
                xr -= xi * tc::pio2_3<float>();
                int n = static_cast<int>(xi);
                quadrant = n & 3;
                return xr;
            }
            else if (x <= tc::mediumpi<float>())
            {
                float fn = std::nearbyint(x * tc::twoopi<float>());
                float r = x - fn * tc::pio2_1<float>();
                float w = fn * tc::pio2_1t<float>();
                float t = r;
                w = fn * tc::pio2_2<float>();
                r = t - w;
                w = fn * tc::pio2_2t<float>() - ((t - r) - w);
                t = r;
                w = fn * tc::pio2_3<float>();
                r = t - w;
                w = fn * tc::pio2_3t<float>() - ((t - r) - w);
                float xr = r - w;
                int n = static_cast<int>(fn);
                quadrant = n & 3;
                return xr;
            }
            else
            {
                double y[2];
                std::int32_t n = ::xsimd::detail::__ieee754_rem_pio2(static_cast<double>(x), y);
                quadrant = n & 3;
                return static_cast<float>(y[0]);
            }
        }

        template <>
        XSIMD_INLINE double trigo_reduce_scalar<double>(double x, int& quadrant) noexcept
        {
            namespace tc = xsimd::constants;

            if (x <= tc::pio4<double>())
            {
                quadrant = 0;
                return x;
            }
            else if (x <= tc::pio2<double>())
            {
                quadrant = (x > tc::pio4<double>()) ? 1 : 0;
                double xr = x - tc::pio2_1<double>();
                xr -= tc::pio2_2<double>();
                xr -= tc::pio2_3<double>();
                return quadrant ? xr : x;
            }
            else if (x <= tc::twentypi<double>())
            {
                double xi = std::nearbyint(x * tc::twoopi<double>());
                double xr = -xi * tc::pio2_1<double>() + x;
                xr -= xi * tc::pio2_2<double>();
                xr -= xi * tc::pio2_3<double>();
                int n = static_cast<int>(xi);
                quadrant = n & 3;
                return xr;
            }
            else if (x <= tc::mediumpi<double>())
            {
                double fn = std::nearbyint(x * tc::twoopi<double>());
                double r = x - fn * tc::pio2_1<double>();
                double w = fn * tc::pio2_1t<double>();
                double t = r;
                w = fn * tc::pio2_2<double>();
                r = t - w;
                w = fn * tc::pio2_2t<double>() - ((t - r) - w);
                t = r;
                w = fn * tc::pio2_3<double>();
                r = t - w;
                w = fn * tc::pio2_3t<double>() - ((t - r) - w);
                double xr = r - w;
                int n = static_cast<int>(fn);
                quadrant = n & 3;
                return xr;
            }
            else
            {
                double y[2];
                std::int32_t n = ::xsimd::detail::__ieee754_rem_pio2(x, y);
                quadrant = n & 3;
                return y[0];
            }
        }

        template <class T>
        XSIMD_INLINE std::pair<T, T> scalar_sincos_impl(T val) noexcept
        {
            T x = std::fabs(val);
            int quadrant = 0;
            T xr = trigo_reduce_scalar<T>(x, quadrant);

            T tmp = (quadrant >= 2) ? T(1.) : T(0.);
            int swap_bit = quadrant - static_cast<int>(2.f * tmp);
            T signmask = ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(
                sizeof(T) == sizeof(float) ? uint32_t(0x80000000u) : uint64_t(0x8000000000000000ull)));

            T z = xr * xr;
            T se = scalar_sin_eval(z, xr);
            T ce = scalar_cos_eval(z);

            T sin_sign = ::xsimd::bitwise_xor(::xsimd::bitwise_and(val, signmask), (tmp != T(0.) ? signmask : T(0.)));
            T cos_sign = ((swap_bit ^ static_cast<int>(tmp)) != 0) ? signmask : T(0.);

            T sin_res = (swap_bit == 0) ? se : ce;
            T cos_res = (swap_bit != 0) ? se : ce;

            sin_res = ::xsimd::bitwise_xor(sin_res, sin_sign);
            cos_res = ::xsimd::bitwise_xor(cos_res, cos_sign);
            return std::make_pair(sin_res, cos_res);
        }
    }

    XSIMD_INLINE float sin(float val) noexcept
    {
        return detail::scalar_sincos_impl<float>(val).first;
    }

    XSIMD_INLINE double sin(double val) noexcept
    {
        return detail::scalar_sincos_impl<double>(val).first;
    }

    XSIMD_INLINE float cos(float val) noexcept
    {
        return detail::scalar_sincos_impl<float>(val).second;
    }

    XSIMD_INLINE double cos(double val) noexcept
    {
        return detail::scalar_sincos_impl<double>(val).second;
    }

    XSIMD_INLINE std::pair<float, float> sincos(float val) noexcept
    {
        return detail::scalar_sincos_impl<float>(val);
    }

    XSIMD_INLINE std::pair<double, double> sincos(double val) noexcept
    {
        return detail::scalar_sincos_impl<double>(val);
    }

    namespace detail
    {
        template <class T>
        XSIMD_INLINE T horner1_scalar(T x) noexcept
        {
            return T(1.);
        }

        template <class T, uint64_t c0>
        XSIMD_INLINE T horner1_scalar(T x) noexcept
        {
            return x + ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(c0));
        }

        template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
        XSIMD_INLINE T horner1_scalar(T x) noexcept
        {
            return x * horner1_scalar<T, c1, args...>(x) + ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(c0));
        }

        XSIMD_INLINE float scalar_base_tancot_eval(float z) noexcept
        {
            float zz = z * z;
            float y = horner_scalar<float,
                                    0x3eaaaa6f,
                                    0x3e0896dd,
                                    0x3d5ac5c9,
                                    0x3cc821b5,
                                    0x3b4c779c,
                                    0x3c19c53b>(zz);
            return y * (zz * z) + z;
        }

        XSIMD_INLINE double scalar_base_tancot_eval(double z) noexcept
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
            return z * (zz * (num / den)) + z;
        }

        template <class T>
        XSIMD_INLINE T scalar_tan_eval(T z, bool test) noexcept
        {
            T y = scalar_base_tancot_eval(z);
            return test ? y : -T(1.) / y;
        }

        template <class T>
        XSIMD_INLINE T scalar_cot_eval(T z, bool test) noexcept
        {
            T y = scalar_base_tancot_eval(z);
            return test ? T(1.) / y : -y;
        }

        XSIMD_INLINE float scalar_sinh_kernel(float x) noexcept
        {
            float sqr = x * x;
            float p = horner_scalar<float,
                                     0x3f800000,
                                     0x3e2aaacc,
                                     0x3c087bbe,
                                     0x39559e2f>(sqr);
            return p * x;
        }

        XSIMD_INLINE double scalar_sinh_kernel(double x) noexcept
        {
            double sqr = x * x;
            double num = horner_scalar<double,
                                       0xc115782bdbf6ab05ull,
                                       0xc0c694b8c71d6182ull,
                                       0xc064773a398ff4feull,
                                       0xbfe9435fe8bb3cd6ull>(sqr);
            double den = horner1_scalar<double,
                                        0xc1401a20e4f90044ull,
                                        0x40e1a7ba7ed72245ull,
                                        0xc0715b6096e96484ull>(sqr);
            return x * ((num / den) * sqr) + x;
        }

        template <class T>
        XSIMD_INLINE T scalar_sinh_impl(T val) noexcept
        {
            namespace tc = xsimd::constants;
            T x = std::fabs(val);
            T r;
            if (x < T(1.))
            {
                r = scalar_sinh_kernel(x);
            }
            else if (x > tc::maxlog<T>() - tc::log_2<T>())
            {
                T half_x = x * T(0.5);
                T tmp = std::exp(half_x);
                r = T(0.5) * tmp * tmp;
            }
            else
            {
                T tmp = std::exp(x);
                r = T(0.5) * tmp - T(0.5) / tmp;
            }
            T signmask = ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(
                sizeof(T) == sizeof(float) ? uint32_t(0x80000000u) : uint64_t(0x8000000000000000ull)));
            return ::xsimd::bitwise_xor(r, ::xsimd::bitwise_and(val, signmask));
        }

        template <class T>
        XSIMD_INLINE T scalar_cosh_impl(T val) noexcept
        {
            namespace tc = xsimd::constants;
            T x = std::fabs(val);
            if (x > tc::maxlog<T>() - tc::log_2<T>())
            {
                T half_x = x * T(0.5);
                T tmp = std::exp(half_x);
                return T(0.5) * tmp * tmp;
            }
            else
            {
                T tmp = std::exp(x);
                return T(0.5) * (tmp + T(1.) / tmp);
            }
        }

        // Scalar ports of the vector log / log1p kernels. They are used by the
        // scalar asinh / acosh paths so that those functions produce results
        // that are bit-identical to their vector counterparts (std::log and
        // std::log1p use different algorithms and would diverge by ~1 ULP).
        //
        // The arithmetic mirrors the vector kernels exactly: fma(a,b,c) is
        // expressed as a*b+c (matching the SSE2 fallback of xsimd::fma), and
        // the same horner coefficients are reused via horner_scalar.
        XSIMD_INLINE float scalar_log(float self) noexcept
        {
            namespace tc = ::xsimd::constants;
            using i_type = std::int32_t;
            float x = self;
            i_type k = 0;
            bool isnez = (self != 0.f);
#ifndef XSIMD_NO_DENORMALS
            if (isnez && self < tc::smallestposval<float>())
            {
                k -= 23;
                x = x * float(8388608ul);
            }
#endif
            i_type ix = ::xsimd::bit_cast<i_type>(x);
            ix += 0x3f800000 - 0x3f3504f3;
            k += (ix >> 23) - 0x7f;
            ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
            x = ::xsimd::bit_cast<float>(ix);
            float f = --x;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t2 + t1;
            float hfsq = 0.5f * f * f;
            float dk = static_cast<float>(k);
            float r = dk * tc::log_2hi<float>() + (s * (hfsq + R) + dk * tc::log_2lo<float>() - hfsq + f);
            if (isnez)
            {
                return (self == tc::infinity<float>()) ? tc::infinity<float>() : r;
            }
            return tc::minusinfinity<float>();
        }

        XSIMD_INLINE double scalar_log(double self) noexcept
        {
            namespace tc = ::xsimd::constants;
            using i_type = std::int64_t;
            double x = self;
            i_type k = 0;
            bool isnez = (self != 0.);
#ifndef XSIMD_NO_DENORMALS
            if (isnez && self < tc::smallestposval<double>())
            {
                k -= 54;
                x = x * double(18014398509481984ull);
            }
#endif
            i_type hx = ::xsimd::bit_cast<i_type>(x) >> 32;
            hx += 0x3ff00000 - 0x3fe6a09e;
            k += (hx >> 20) - 0x3ff;
            double dk = static_cast<double>(k);
            hx = (hx & i_type(0x000fffff)) + 0x3fe6a09e;
            i_type low = ::xsimd::bit_cast<i_type>(x) & i_type(0xffffffff);
            x = ::xsimd::bit_cast<double>((hx << 32) | low);
            double f = --x;
            double hfsq = 0.5 * f * f;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;
            double t1 = w * horner_scalar<double, 0x3fd999999997fa04ull, 0x3fcc71c51d8e78afull, 0x3fc39a09d078c69full>(w);
            double t2 = z * horner_scalar<double, 0x3fe5555555555593ull, 0x3fd2492494229359ull, 0x3fc7466496cb03deull, 0x3fc2f112df3e5244ull>(w);
            double R = t2 + t1;
            double r = dk * tc::log_2hi<double>() + (s * (hfsq + R) + dk * tc::log_2lo<double>() - hfsq + f);
            if (isnez)
            {
                return (self == tc::infinity<double>()) ? tc::infinity<double>() : r;
            }
            return tc::minusinfinity<double>();
        }

        XSIMD_INLINE float scalar_log1p(float self) noexcept
        {
            namespace tc = ::xsimd::constants;
            using i_type = std::int32_t;
            float uf = self + 1.f;
            bool isnez = (uf != 0.f);
            i_type iu = ::xsimd::bit_cast<i_type>(uf);
            iu += 0x3f800000 - 0x3f3504f3;
            i_type k = (iu >> 23) - 0x7f;
            iu = (iu & i_type(0x007fffff)) + 0x3f3504f3;
            float f = ::xsimd::bit_cast<float>(iu);
            f = f - 1.f;
            float s = f / (2.f + f);
            float z = s * s;
            float w = z * z;
            float t1 = w * horner_scalar<float, 0x3eccce13, 0x3e789e26>(w);
            float t2 = z * horner_scalar<float, 0x3f2aaaaa, 0x3e91e9ee>(w);
            float R = t2 + t1;
            float hfsq = 0.5f * f * f;
            float dk = static_cast<float>(k);
            float c = (k >= i_type(2)) ? (1.f - (uf - self)) : (self - (uf - 1.f));
            c = c / uf;
            float r = dk * tc::log_2hi<float>() + (s * (hfsq + R) + dk * tc::log_2lo<float>() + c - hfsq + f);
            if (isnez)
            {
                return (self == tc::infinity<float>()) ? tc::infinity<float>() : r;
            }
            return tc::minusinfinity<float>();
        }

        XSIMD_INLINE double scalar_log1p(double self) noexcept
        {
            namespace tc = ::xsimd::constants;
            using i_type = std::int64_t;
            double uf = self + 1.;
            bool isnez = (uf != 0.);
            i_type hu = ::xsimd::bit_cast<i_type>(uf) >> 32;
            hu += 0x3ff00000 - 0x3fe6a09e;
            i_type k = (hu >> 20) - 0x3ff;
            double c = (k >= i_type(2)) ? (1. - (uf - self)) : (self - (uf - 1.));
            c = c / uf;
            hu = (hu & i_type(0x000fffff)) + 0x3fe6a09e;
            i_type low = ::xsimd::bit_cast<i_type>(uf) & i_type(0xffffffff);
            double f = ::xsimd::bit_cast<double>((hu << 32) | low);
            f = --f;
            double hfsq = 0.5 * f * f;
            double s = f / (2. + f);
            double z = s * s;
            double w = z * z;
            double t1 = w * horner_scalar<double, 0x3fd999999997fa04ull, 0x3fcc71c51d8e78afull, 0x3fc39a09d078c69full>(w);
            double t2 = z * horner_scalar<double, 0x3fe5555555555593ull, 0x3fd2492494229359ull, 0x3fc7466496cb03deull, 0x3fc2f112df3e5244ull>(w);
            double R = t2 + t1;
            double dk = static_cast<double>(k);
            double r = dk * tc::log_2hi<double>() + (s * (hfsq + R) + dk * tc::log_2lo<double>() + c - hfsq + f);
            if (isnez)
            {
                return (self == tc::infinity<double>()) ? tc::infinity<double>() : r;
            }
            return tc::minusinfinity<double>();
        }

        XSIMD_INLINE float scalar_asin(float self) noexcept
        {
            namespace tc = xsimd::constants;
            float x = std::fabs(self);
            bool x_larger_05 = x > 0.5f;
            float z = x_larger_05 ? 0.5f * (1.f - x) : x * x;
            x = x_larger_05 ? std::sqrt(z) : x;
            float z1 = horner_scalar<float,
                                     0x3e2aaae4,
                                     0x3d9980f6,
                                     0x3d3a3ec7,
                                     0x3cc617e3,
                                     0x3d2cb352>(z);
            z1 = z1 * (z * x) + x;
            z = x_larger_05 ? tc::pio2<float>() - (z1 + z1) : z1;
            return std::copysign(z, self);
        }

        XSIMD_INLINE double scalar_asin(double self) noexcept
        {
            namespace tc = xsimd::constants;
            double x = std::fabs(self);
            bool small_cond = x < tc::sqrteps<double>();
            double ct1 = ::xsimd::bit_cast<double>(int64_t(0x3fe4000000000000));
            double zz1 = 1. - x;
            double vp = zz1 * horner_scalar<double, 0x403c896240f3081dull, 0xc03991aaac01ab68ull, 0x401bdff5baf33e6aull, 0xbfe2079259f9290full, 0x3f684fc3988e9f08ull>(zz1) / horner1_scalar<double, 0x40756709b0b644beull, 0xc077fe08959063eeull, 0x40626219af6a7f42ull, 0xc035f2a2b6bf5d8cull>(zz1);
            zz1 = std::sqrt(zz1 + zz1);
            double z = tc::pio4<double>() - zz1;
            zz1 = zz1 * vp - tc::pio_2lo<double>();
            z = z - zz1;
            zz1 = z + tc::pio4<double>();
            double zz2 = self * self;
            z = zz2 * horner_scalar<double, 0xc020656c06ceafd5ull, 0x40339007da779259ull, 0xc0304331de27907bull, 0x4015c74b178a2dd9ull, 0xbfe34341333e5c16ull, 0x3f716b9b0bd48ad3ull>(zz2) / horner1_scalar<double, 0xc04898220a3607acull, 0x4061705684ffbf9dull, 0xc06265bb6d3576d7ull, 0x40519fc025fe9054ull, 0xc02d7b590b5e0eabull>(zz2);
            zz2 = x * z + x;
            if (x > 1.)
                return tc::nan<double>();
            double res = small_cond ? x : ((x > ct1) ? zz1 : zz2);
            return std::copysign(res, self);
        }

        template <class T>
        XSIMD_INLINE T scalar_asin(T self) noexcept
        {
            return std::asin(self);
        }

        template <class T>
        XSIMD_INLINE T scalar_acos(T self) noexcept
        {
            T x = std::fabs(self);
            bool x_larger_05 = x > T(0.5);
            x = x_larger_05 ? std::sqrt(-T(0.5) * x + T(0.5)) : self;
            x = scalar_asin(x);
            x = x_larger_05 ? x + x : x;
            x = (self < T(-0.5)) ? (constants::pi<T>() - x) : x;
            return x_larger_05 ? x : (constants::pio2<T>() - x);
        }

        XSIMD_INLINE float scalar_asinh(float self) noexcept
        {
            namespace tc = xsimd::constants;
            float x = std::fabs(self);
            bool lthalf = x < 0.5f;
            float x2 = x * x;
            float z = 0.f;
            if (lthalf)
            {
                z = horner_scalar<float,
                                  0x3f800000,
                                  0xbe2aa9ad,
                                  0x3d9949b1,
                                  0xbd2ee581,
                                  0x3ca4d6e6>(x2)
                    * x;
                if (lthalf)
                    return std::copysign(z, self);
            }
            float h = std::sqrt(1.f + x * x);
            float tmp = (x > tc::oneosqrteps<float>()) ? x : (x * 0.5f + h * 0.5f);
#ifndef XSIMD_NO_NANS
            if (std::isnan(self))
                return tc::nan<float>();
#endif
            return std::copysign(scalar_log(tmp) + tc::log_2<float>(), self);
        }

        XSIMD_INLINE double scalar_asinh(double self) noexcept
        {
            namespace tc = xsimd::constants;
            double x = std::fabs(self);
            bool test = x > tc::oneosqrteps<double>();
            double z = test ? x - 1. : x + x * x / (1. + std::hypot(1., x));
#ifndef XSIMD_NO_INFINITIES
            z = (x == tc::infinity<double>()) ? x : z;
#endif
            double l1pz = scalar_log1p(z);
            z = test ? l1pz + tc::log_2<double>() : l1pz;
            return std::copysign(z, self);
        }

        template <class T>
        XSIMD_INLINE T scalar_asinh(T self) noexcept
        {
            return std::asinh(self);
        }

        template <class T>
        XSIMD_INLINE T scalar_acosh(T self) noexcept
        {
            namespace tc = xsimd::constants;
            T x = self - T(1.);
            bool test = x > tc::oneotwoeps<T>();
            T z = test ? self : (x + std::sqrt(x + x + x * x));
            T l1pz = scalar_log1p(z);
            return test ? l1pz + tc::log_2<T>() : l1pz;
        }

        template <class T>
        XSIMD_INLINE T scalar_tan_impl(T val) noexcept
        {
            T x = std::fabs(val);
            int quadrant = 0;
            T xr = trigo_reduce_scalar<T>(x, quadrant);
            T tmp = (quadrant >= 2) ? T(1.) : T(0.);
            int swap_bit = quadrant - static_cast<int>(2.f * tmp);
            bool test = (swap_bit == 0);
            T y = scalar_tan_eval(xr, test);
            T signmask = ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(
                sizeof(T) == sizeof(float) ? uint32_t(0x80000000u) : uint64_t(0x8000000000000000ull)));
            return ::xsimd::bitwise_xor(y, ::xsimd::bitwise_and(val, signmask));
        }

        template <class T>
        XSIMD_INLINE T scalar_cot_impl(T val) noexcept
        {
            T x = std::fabs(val);
            int quadrant = 0;
            T xr = trigo_reduce_scalar<T>(x, quadrant);
            T tmp = (quadrant >= 2) ? T(1.) : T(0.);
            int swap_bit = quadrant - static_cast<int>(2.f * tmp);
            bool test = (swap_bit == 0);
            T y = scalar_cot_eval(xr, test);
            T signmask = ::xsimd::bit_cast<T>(static_cast<::xsimd::as_unsigned_integer_t<T>>(
                sizeof(T) == sizeof(float) ? uint32_t(0x80000000u) : uint64_t(0x8000000000000000ull)));
            return ::xsimd::bitwise_xor(y, ::xsimd::bitwise_and(val, signmask));
        }

        XSIMD_INLINE float scalar_atan(float self) noexcept
        {
            namespace tc = xsimd::constants;
            float x = std::fabs(self);
            float recx = 1.f / x;
            bool flag1 = x < tc::tan3pio8<float>();
            bool flag2 = (x >= ::xsimd::bit_cast<float>(0x3ed413cdu)) && flag1;
            float yy = flag1 ? 0.f : tc::pio2<float>();
            yy = flag2 ? tc::pio4<float>() : yy;
            float xx = flag1 ? x : -recx;
            xx = flag2 ? (x - 1.f) / (x + 1.f) : xx;
            float z = xx * xx;
            float z1 = horner_scalar<float,
                                     0xbeaaaa2a,
                                     0x3e4c925f,
                                     0xbe0e1b85,
                                     0x3da4f0d1>(z);
            z1 = xx * (z1 * z) + xx;
            z1 = flag2 ? z1 + tc::pio_4lo<float>() : z1;
            z1 = !flag1 ? z1 + tc::pio_2lo<float>() : z1;
            float res = yy + z1;
            return std::copysign(res, self);
        }

        XSIMD_INLINE double scalar_atan(double self) noexcept
        {
            namespace tc = xsimd::constants;
            double x = std::fabs(self);
            double recx = 1. / x;
            bool flag1 = x < tc::tan3pio8<double>();
            bool flag2 = (x >= tc::tanpio8<double>()) && flag1;
            double yy = flag1 ? 0. : tc::pio2<double>();
            yy = flag2 ? tc::pio4<double>() : yy;
            double xx = flag1 ? x : -recx;
            xx = flag2 ? (x - 1.) / (x + 1.) : xx;
            double z = xx * xx;
            z *= horner_scalar<double,
                               0xc0503669fd28ec8eull,
                               0xc05eb8bf2d05ba25ull,
                               0xc052c08c36880273ull,
                               0xc03028545b6b807aull,
                               0xbfec007fa1f72594ull>(z)
                / horner1_scalar<double,
                                 0x4068519efbbd62ecull,
                                 0x407e563f13b049eaull,
                                 0x407b0e18d2e2be3bull,
                                 0x4064a0dd43b8fa25ull,
                                 0x4038dbc45b14603cull>(z);
            z = xx * z + xx;
            z = flag2 ? z + tc::pio_4lo<double>() : z;
            z = z + (flag1 ? 0. : tc::pio_2lo<double>());
            double res = yy + z;
            return std::copysign(res, self);
        }

        template <class T>
        XSIMD_INLINE T scalar_atan(T self) noexcept
        {
            return std::atan(self);
        }

        template <class T>
        XSIMD_INLINE T scalar_acot(T self) noexcept
        {
            return constants::pio2<T>() - scalar_atan(self);
        }
    }

    XSIMD_INLINE float sinh(float val) noexcept
    {
        return detail::scalar_sinh_impl(val);
    }

    XSIMD_INLINE double sinh(double val) noexcept
    {
        return detail::scalar_sinh_impl(val);
    }

    XSIMD_INLINE float cosh(float val) noexcept
    {
        return detail::scalar_cosh_impl(val);
    }

    XSIMD_INLINE double cosh(double val) noexcept
    {
        return detail::scalar_cosh_impl(val);
    }

    XSIMD_INLINE float asin(float val) noexcept
    {
        return detail::scalar_asin(val);
    }

    XSIMD_INLINE double asin(double val) noexcept
    {
        return detail::scalar_asin(val);
    }

    XSIMD_INLINE float acos(float val) noexcept
    {
        return detail::scalar_acos(val);
    }

    XSIMD_INLINE double acos(double val) noexcept
    {
        return detail::scalar_acos(val);
    }

    XSIMD_INLINE float asinh(float val) noexcept
    {
        return detail::scalar_asinh(val);
    }

    XSIMD_INLINE double asinh(double val) noexcept
    {
        return detail::scalar_asinh(val);
    }

    XSIMD_INLINE float acosh(float val) noexcept
    {
        return detail::scalar_acosh(val);
    }

    XSIMD_INLINE double acosh(double val) noexcept
    {
        return detail::scalar_acosh(val);
    }


    XSIMD_INLINE float tan(float val) noexcept
    {
        return detail::scalar_tan_impl(val);
    }

    XSIMD_INLINE double tan(double val) noexcept
    {
        return detail::scalar_tan_impl(val);
    }

    XSIMD_INLINE float cot(float val) noexcept
    {
        return detail::scalar_cot_impl(val);
    }

    XSIMD_INLINE double cot(double val) noexcept
    {
        return detail::scalar_cot_impl(val);
    }

    XSIMD_INLINE float atan(float val) noexcept
    {
        return detail::scalar_atan(val);
    }

    XSIMD_INLINE double atan(double val) noexcept
    {
        return detail::scalar_atan(val);
    }

    XSIMD_INLINE float acot(float val) noexcept
    {
        return detail::scalar_acot(val);
    }

    XSIMD_INLINE double acot(double val) noexcept
    {
        return detail::scalar_acot(val);
    }

    template <class T>
    XSIMD_INLINE std::complex<T> sin(const std::complex<T>& z) noexcept
    {
        return std::complex<T>(sin(z.real()) * cosh(z.imag()),
                               cos(z.real()) * sinh(z.imag()));
    }

    template <class T>
    XSIMD_INLINE std::complex<T> cos(const std::complex<T>& z) noexcept
    {
        return std::complex<T>(cos(z.real()) * cosh(z.imag()),
                               -sin(z.real()) * sinh(z.imag()));
    }

    template <class T>
    XSIMD_INLINE std::complex<T> sinh(const std::complex<T>& z) noexcept
    {
        return std::complex<T>(sinh(z.real()) * cos(z.imag()),
                               cosh(z.real()) * sin(z.imag()));
    }

    template <class T>
    XSIMD_INLINE std::complex<T> cosh(const std::complex<T>& z) noexcept
    {
        return std::complex<T>(cosh(z.real()) * cos(z.imag()),
                               sinh(z.real()) * sin(z.imag()));
    }

    // Complex scalar overloads for the inverse trig / hyperbolic functions.
    // The xsimd vector kernels use a different algorithm (log/sqrt based) than
    // the standard library, so delegating to std::xxx would diverge by many
    // ULPs. To guarantee bit-identical results between the scalar complex
    // overloads and the complex batch, the scalar overloads route the value
    // through the exact same vector kernel (a single-lane batch) and extract
    // the lane back.
    namespace detail
    {
        template <class T, class Fn>
        XSIMD_INLINE std::complex<T> complex_scalar_via_batch(const std::complex<T>& z, Fn fn) noexcept
        {
            using cbatch = batch<std::complex<T>, default_arch>;
            cbatch in(z);
            cbatch out = fn(in);
            return out.get(0);
        }
    }

    template <class T>
    XSIMD_INLINE std::complex<T> asin(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::asin(b); });
    }

    template <class T>
    XSIMD_INLINE std::complex<T> acos(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::acos(b); });
    }

    template <class T>
    XSIMD_INLINE std::complex<T> asinh(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::asinh(b); });
    }

    template <class T>
    XSIMD_INLINE std::complex<T> acosh(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::acosh(b); });
    }

    template <class T>
    XSIMD_INLINE std::complex<T> atan(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::atan(b); });
    }

    template <class T>
    XSIMD_INLINE std::complex<T> tan(const std::complex<T>& z) noexcept
    {
        return detail::complex_scalar_via_batch<T>(z, [](const batch<std::complex<T>, default_arch>& b) { return ::xsimd::tan(b); });
    }
}
#endif
