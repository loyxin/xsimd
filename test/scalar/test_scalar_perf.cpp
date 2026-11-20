/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// Performance comparison: xsimd::scalar kernels vs the system libm.
//
// The timing harness feeds each function a cycling sequence of input
// values so the compiler cannot constant-fold the calls, and accumulates
// the results into a sink that is printed at the end (so the calls
// cannot be dead-code eliminated). For every function we report
//
//   * xsimd   : ns/call of the xsimd::scalar kernel
//   * libm    : ns/call of the system libm function
//   * speedup : libm / xsimd (> 1 means xsimd is faster)
//
// Inputs are plain finite values in the "hot" range of each function;
// this is a throughput micro-benchmark, not a special-case study.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "xsimd/xsimd.hpp"

namespace
{
    using clock_type = std::chrono::steady_clock;

    // Number of inner repetitions per measurement; the reported time is
    // the best of several rounds to damp scheduler noise.
    constexpr int reps = 200000;
    constexpr int rounds = 7;

    // Cheap pseudo-random but fully deterministic input generator
    // (xorshift32), so both sides are fed exactly the same values.
    struct input_sequence
    {
        uint32_t state;
        explicit input_sequence(uint32_t seed)
            : state(seed ? seed : 1u)
        {
        }
        double next(double lo, double hi)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return lo + (double(state) / double(0xffffffffu)) * (hi - lo);
        }
    };

    template <class F>
    double best_ns_per_call(F&& fn)
    {
        double best = 1e30;
        for (int r = 0; r < rounds; ++r)
        {
            auto t0 = clock_type::now();
            double sink = fn();
            auto t1 = clock_type::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
            if (ns < best)
                best = ns;
            if (sink == 12345.678e300) // never true, keeps the sink alive
                std::printf("%f", sink);
        }
        return best;
    }

    // Sink printed once at the end so calls are not eliminated.
    double g_sink = 0.0;

    struct result_row
    {
        const char* name;
        double xsimd_ns;
        double libm_ns;
    };

    std::vector<result_row> g_rows;

    template <class XS, class Libm>
    void benchmark(const char* name, double lo, double hi, XS&& xs, Libm&& lm)
    {
        // identical inputs for both sides
        std::vector<double> vals(reps);
        input_sequence seq(0x9e3779b9u);
        for (int i = 0; i < reps; ++i)
            vals[i] = seq.next(lo, hi);

        double sx = best_ns_per_call([&]
                                     {
            double acc = 0.0;
            for (int i = 0; i < reps; ++i)
                acc += xs(vals[i]);
            g_sink += acc;
            return acc; });
        double sl = best_ns_per_call([&]
                                     {
            double acc = 0.0;
            for (int i = 0; i < reps; ++i)
                acc += lm(vals[i]);
            g_sink += acc;
            return acc; });
        g_rows.push_back({ name, sx, sl });
    }
}

TEST_CASE("performance vs libm")
{
    // "hot" input range per function, mirroring typical benchmark suites
    benchmark("sin", -3.14159265, 3.14159265,
              [](double x)
              { return xsimd::scalar::sin(x); },
              [](double x)
              { return std::sin(x); });
    benchmark("cos", -3.14159265, 3.14159265,
              [](double x)
              { return xsimd::scalar::cos(x); },
              [](double x)
              { return std::cos(x); });
    benchmark("tan", -1.5, 1.5,
              [](double x)
              { return xsimd::scalar::tan(x); },
              [](double x)
              { return std::tan(x); });
    benchmark("asin", -1.0, 1.0,
              [](double x)
              { return xsimd::scalar::asin(x); },
              [](double x)
              { return std::asin(x); });
    benchmark("acos", -1.0, 1.0,
              [](double x)
              { return xsimd::scalar::acos(x); },
              [](double x)
              { return std::acos(x); });
    benchmark("atan", -2.0, 2.0,
              [](double x)
              { return xsimd::scalar::atan(x); },
              [](double x)
              { return std::atan(x); });
    benchmark("atan2", 0.0, 1.0,
              [](double x)
              { return xsimd::scalar::atan2(x, 0.7); },
              [](double x)
              { return std::atan2(x, 0.7); });
    benchmark("sinh", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::sinh(x); },
              [](double x)
              { return std::sinh(x); });
    benchmark("cosh", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::cosh(x); },
              [](double x)
              { return std::cosh(x); });
    benchmark("tanh", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::tanh(x); },
              [](double x)
              { return std::tanh(x); });
    benchmark("asinh", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::asinh(x); },
              [](double x)
              { return std::asinh(x); });
    benchmark("acosh", 1.0, 3.0,
              [](double x)
              { return xsimd::scalar::acosh(x); },
              [](double x)
              { return std::acosh(x); });
    benchmark("atanh", -0.9, 0.9,
              [](double x)
              { return xsimd::scalar::atanh(x); },
              [](double x)
              { return std::atanh(x); });
    benchmark("exp", -20.0, 20.0,
              [](double x)
              { return xsimd::scalar::exp(x); },
              [](double x)
              { return std::exp(x); });
    benchmark("exp2", -20.0, 20.0,
              [](double x)
              { return xsimd::scalar::exp2(x); },
              [](double x)
              { return std::exp2(x); });
    benchmark("expm1", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::expm1(x); },
              [](double x)
              { return std::expm1(x); });
    benchmark("log", 1e-10, 100.0,
              [](double x)
              { return xsimd::scalar::log(x); },
              [](double x)
              { return std::log(x); });
    benchmark("log2", 1e-10, 100.0,
              [](double x)
              { return xsimd::scalar::log2(x); },
              [](double x)
              { return std::log2(x); });
    benchmark("log10", 1e-10, 100.0,
              [](double x)
              { return xsimd::scalar::log10(x); },
              [](double x)
              { return std::log10(x); });
    benchmark("log1p", -0.99, 3.0,
              [](double x)
              { return xsimd::scalar::log1p(x); },
              [](double x)
              { return std::log1p(x); });
    benchmark("cbrt", 1e-3, 1e3,
              [](double x)
              { return xsimd::scalar::cbrt(x); },
              [](double x)
              { return std::cbrt(x); });
    benchmark("hypot", 0.0, 1.0,
              [](double x)
              { return xsimd::scalar::hypot(x, 0.6); },
              [](double x)
              { return std::hypot(x, 0.6); });
    benchmark("erf", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::erf(x); },
              [](double x)
              { return std::erf(x); });
    benchmark("erfc", -3.0, 3.0,
              [](double x)
              { return xsimd::scalar::erfc(x); },
              [](double x)
              { return std::erfc(x); });
    benchmark("lgamma", 0.1, 20.0,
              [](double x)
              { return xsimd::scalar::lgamma(x); },
              [](double x)
              { return std::lgamma(x); });
    benchmark("tgamma", 0.1, 20.0,
              [](double x)
              { return xsimd::scalar::tgamma(x); },
              [](double x)
              { return std::tgamma(x); });

    // ---- report ----
    std::printf("\n%-10s %12s %12s %9s\n", "function", "xsimd[ns]", "libm[ns]", "speedup");
    std::printf("%-10s %12s %12s %9s\n", "--------", "---------", "--------", "-------");
    for (const auto& r : g_rows)
    {
        double speedup = r.xsimd_ns > 0.0 ? r.libm_ns / r.xsimd_ns : 0.0;
        std::printf("%-10s %12.2f %12.2f %9.2fx\n",
                    r.name, r.xsimd_ns, r.libm_ns, speedup);
    }
    std::printf("\n(sink = %.17g)\n", g_sink);
}
