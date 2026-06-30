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

#include "xsimd/xsimd.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    // Fixed seed Mersenne Twister for reproducible random data.
    constexpr std::uint32_t fixed_seed = 20260630u;
    constexpr size_t nb_samples = 10000;

    // FNV-1a 64-bit hash over a raw byte buffer.
    std::uint64_t fnv1a_hash(const void* data, std::size_t size) noexcept
    {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i)
        {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    // Signed ULP distance between two finite floats (a-b measured in units of
    // the least significant bit of their common magnitude). Used only for
    // diagnostics so we can see how far off a mismatching lane really is.
    template <class T>
    std::int64_t ulp_distance(T a, T b) noexcept
    {
        static_assert(std::is_floating_point<T>::value, "T must be floating point");
        using bits_t = typename std::conditional<sizeof(T) == 4, std::int32_t, std::int64_t>::type;
        bits_t ia, ib;
        std::memcpy(&ia, &a, sizeof(T));
        std::memcpy(&ib, &b, sizeof(T));
        if (ia < 0) ia = std::numeric_limits<bits_t>::min() - ia;
        if (ib < 0) ib = std::numeric_limits<bits_t>::min() - ib;
        return static_cast<std::int64_t>(ia) - static_cast<std::int64_t>(ib);
    }

    template <class T>
    std::string hex_bits(T v) noexcept
    {
        using bits_t = typename std::conditional<sizeof(T) == 4, std::uint32_t, std::uint64_t>::type;
        bits_t i;
        std::memcpy(&i, &v, sizeof(T));
        std::ostringstream ss;
        ss << "0x" << std::hex << std::setfill('0') << std::setw(sizeof(T) * 2) << i;
        return ss.str();
    }

    // Generate nb_samples random values in [lo, hi).
    template <class T>
    std::vector<T> make_uniform(T lo, T hi)
    {
        std::mt19937 gen(fixed_seed);
        std::uniform_real_distribution<T> dist(lo, hi);
        std::vector<T> data(nb_samples);
        for (size_t i = 0; i < nb_samples; ++i)
        {
            data[i] = dist(gen);
        }
        return data;
    }

    // Core check: for every sample, the scalar implementation must produce the
    // same bits as the corresponding lane of the vector implementation.
    //
    // The scalar path and the vector kernel share coefficients/algorithms but
    // operate on independent lanes; bit-identical lane results imply the two
    // paths are algorithmically equivalent.
    template <class B, class ScalarFn, class VectorFn>
    std::uint64_t check_scalar_eq_vector(const std::vector<typename B::value_type>& input,
                                ScalarFn scalar_fn, VectorFn vector_fn)
    {
        using T = typename B::value_type;
        constexpr size_t N = B::size;
        B in_batch, out_batch;

        std::uint64_t hash = 0xcbf29ce484222325ULL;

        for (size_t i = 0; i + N <= nb_samples; i += N)
        {
            ::detail::load_batch(in_batch, input, i);
            out_batch = vector_fn(in_batch);

            alignas(B::arch_type::alignment()) T lanes[N];
            out_batch.store_aligned(lanes);

            for (size_t lane = 0; lane < N; ++lane)
            {
                const T scalar_val = scalar_fn(input[i + lane]);
                const T vector_val = lanes[lane];
                if (!detail::bit_equal(scalar_val, vector_val))
                {
                    std::ostringstream msg;
                    msg << "lane mismatch at sample " << (i + lane)
                        << " ulp=" << ulp_distance(scalar_val, vector_val)
                        << " scalar=" << std::setprecision(std::numeric_limits<T>::max_digits10) << scalar_val
                        << " (" << hex_bits(scalar_val) << ")"
                        << " vector=" << std::setprecision(std::numeric_limits<T>::max_digits10) << vector_val
                        << " (" << hex_bits(vector_val) << ")"
                        << " input=" << std::setprecision(std::numeric_limits<T>::max_digits10) << input[i + lane];
                    INFO(msg.str());
                    CHECK_UNARY(false);
                }

                const auto* p = reinterpret_cast<const std::uint8_t*>(&scalar_val);
                for (std::size_t b = 0; b < sizeof(T); ++b)
                {
                    hash ^= p[b];
                    hash *= 0x100000001b3ULL;
                }
            }
        }
        return hash;
    }

    // Complex variant: compares the scalar complex overload against the
    // corresponding lane of the complex batch kernel. The scalar complex
    // overloads route through the same vector kernel as the batch, so each
    // component must be bit-identical lane by lane.
    template <class CB, class ScalarFn, class VectorFn>
    std::uint64_t check_scalar_eq_vector_complex(const std::vector<std::complex<typename CB::value_type::value_type>>& input,
                                        ScalarFn scalar_fn, VectorFn vector_fn)
    {
        using CT = typename CB::value_type;            // std::complex<T>
        using T = typename CT::value_type;             // T (float/double)
        constexpr size_t N = CB::size;
        CB in_batch, out_batch;

        std::uint64_t hash = 0xcbf29ce484222325ULL;

        for (size_t i = 0; i + N <= nb_samples; i += N)
        {
            ::detail::load_batch(in_batch, input, i);
            out_batch = vector_fn(in_batch);

            for (size_t lane = 0; lane < N; ++lane)
            {
                const CT scalar_val = scalar_fn(input[i + lane]);
                const CT vector_val = out_batch.get(lane);
                if (!detail::bit_equal(scalar_val.real(), vector_val.real()) ||
                    !detail::bit_equal(scalar_val.imag(), vector_val.imag()))
                {
                    std::ostringstream msg;
                    msg << "complex lane mismatch at sample " << (i + lane)
                        << " scalar=(" << std::setprecision(std::numeric_limits<T>::max_digits10) << scalar_val.real()
                        << "," << scalar_val.imag() << ")"
                        << " vector=(" << vector_val.real()
                        << "," << vector_val.imag() << ")"
                        << " input=(" << input[i + lane].real()
                        << "," << input[i + lane].imag() << ")";
                    INFO(msg.str());
                    CHECK_UNARY(false);
                }

                T re = scalar_val.real();
                T im = scalar_val.imag();
                const auto* pr = reinterpret_cast<const std::uint8_t*>(&re);
                const auto* pi = reinterpret_cast<const std::uint8_t*>(&im);
                for (std::size_t b = 0; b < sizeof(T); ++b)
                {
                    hash ^= pr[b];
                    hash *= 0x100000001b3ULL;
                }
                for (std::size_t b = 0; b < sizeof(T); ++b)
                {
                    hash ^= pi[b];
                    hash *= 0x100000001b3ULL;
                }
            }
        }
        return hash;
    }

    // Generate nb_samples complex values with real/imag parts uniform in [lo, hi).
    template <class T>
    std::vector<std::complex<T>> make_complex_uniform(T lo, T hi)
    {
        std::mt19937 gen(fixed_seed);
        std::uniform_real_distribution<T> dist(lo, hi);
        std::vector<std::complex<T>> data(nb_samples);
        for (size_t i = 0; i < nb_samples; ++i)
        {
            data[i] = std::complex<T>(dist(gen), dist(gen));
        }
        return data;
    }
}

// ---------------------------------------------------------------------------
// Each TEST_CASE covers batch<float> and batch<double> for one function.
// Domains are chosen to match the mathematical domain of each function while
// avoiding values that would overflow the range reduction (e.g. tan near pi/2
// for very large inputs).
// ---------------------------------------------------------------------------

TEST_CASE_TEMPLATE("[scalar vs vector] asin", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-1.0), T(1.0)),
        [](T v) { return ::xsimd::asin(v); },
        [](B const& b) { return ::xsimd::asin(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x17975254C9D5A999ULL);
    else
        CHECK_EQ(h, 0xF37A4C7A301A9AA1ULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] acos", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-1.0), T(1.0)),
        [](T v) { return ::xsimd::acos(v); },
        [](B const& b) { return ::xsimd::acos(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x3517EA2D32EF4272ULL);
    else
        CHECK_EQ(h, 0xE622CC5390BB3414ULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] asinh", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-50.0), T(50.0)),
        [](T v) { return ::xsimd::asinh(v); },
        [](B const& b) { return ::xsimd::asinh(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x42768A221BE33357ULL);
    else
        CHECK_EQ(h, 0x6D5532D57DAE6B34ULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] acosh", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(1.0), T(50.0)),
        [](T v) { return ::xsimd::acosh(v); },
        [](B const& b) { return ::xsimd::acosh(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x3A340D80D8E52982ULL);
    else
        CHECK_EQ(h, 0xE034B3278B72C3DAULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] tan", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-10.0), T(10.0)),
        [](T v) { return ::xsimd::tan(v); },
        [](B const& b) { return ::xsimd::tan(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x0BF8B2E99601BEECULL);
    else
        CHECK_EQ(h, 0x91D5D4019994A8BFULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] cot", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-10.0), T(10.0)),
        [](T v) { return ::xsimd::cot(v); },
        [](B const& b) { return ::xsimd::cot(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x2C59D6B94B26FB82ULL);
    else
        CHECK_EQ(h, 0xEAA5399063EA4810ULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] atan", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-50.0), T(50.0)),
        [](T v) { return ::xsimd::atan(v); },
        [](B const& b) { return ::xsimd::atan(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x2C7FF44C4BDCEFAEULL);
    else
        CHECK_EQ(h, 0x0F6E60A90604427BULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] acot", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    auto h = check_scalar_eq_vector<B>(
        make_uniform<T>(T(-50.0), T(50.0)),
        [](T v) { return ::xsimd::acot(v); },
        [](B const& b) { return ::xsimd::acot(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x3DF081D159880DFDULL);
    else
        CHECK_EQ(h, 0xFEE6D2F4C194F63CULL);
}

// ---------------------------------------------------------------------------
// Complex TEST_CASEs. cot / acot have no complex overloads, so they are not
// tested here. The complex scalar overloads route through the same vector
// kernel as the complex batch, so each component must be bit-identical lane
// by lane (bit-exact equality).
// ---------------------------------------------------------------------------

TEST_CASE_TEMPLATE("[scalar vs vector] complex asin", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::asin(v); },
        [](CB const& b) { return ::xsimd::asin(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x5E4784065DC96889ULL);
    else
        CHECK_EQ(h, 0x9F774362B23E591AULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] complex acos", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::acos(v); },
        [](CB const& b) { return ::xsimd::acos(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x92B5289388232B50ULL);
    else
        CHECK_EQ(h, 0x2EDECAB0B9EDA46EULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] complex asinh", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::asinh(v); },
        [](CB const& b) { return ::xsimd::asinh(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0xC9B66C25AC218268ULL);
    else
        CHECK_EQ(h, 0x3DF112E7820775A7ULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] complex acosh", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::acosh(v); },
        [](CB const& b) { return ::xsimd::acosh(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x359156EC7CB1D4B8ULL);
    else
        CHECK_EQ(h, 0x279473CF4D64DB2EULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] complex atan", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::atan(v); },
        [](CB const& b) { return ::xsimd::atan(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x84A0702B33E235FDULL);
    else
        CHECK_EQ(h, 0xF55332ECBD73046EULL);
}

TEST_CASE_TEMPLATE("[scalar vs vector] complex tan", B, BATCH_FLOAT_TYPES)
{
    using T = typename B::value_type;
    using CB = ::xsimd::batch<std::complex<T>>;
    auto h = check_scalar_eq_vector_complex<CB>(
        make_complex_uniform<T>(T(-5.0), T(5.0)),
        [](std::complex<T> v) { return ::xsimd::tan(v); },
        [](CB const& b) { return ::xsimd::tan(b); });
    if constexpr (sizeof(T) == 4)
        CHECK_EQ(h, 0x1F32D96E5FE9115AULL);
    else
        CHECK_EQ(h, 0xBDD97CFEC9523B16ULL);
}

#endif
