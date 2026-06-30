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

template <class T>
struct scalar_real_trigo_test
{
    using value_type = T;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type ainput;
    vector_type acosh_input;
    vector_type expected;

    scalar_real_trigo_test()
    {
        nb_input = 10000;
        input.resize(nb_input);
        ainput.resize(nb_input);
        acosh_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(0.) + i * value_type(80.) / nb_input;
            ainput[i] = value_type(-1.) + value_type(2.) * i / nb_input;
            acosh_input[i] = value_type(1.) + i * value_type(3.) / nb_input;
        }
        expected.resize(nb_input);
    }

    void test_sin()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sin(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::sin(input[i]);
            INFO("sin");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_cos()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::cos(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::cos(input[i]);
            INFO("cos");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_sincos()
    {
        vector_type expected2(nb_input);
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sin(v); });
        std::transform(input.cbegin(), input.cend(), expected2.begin(),
                       [](const value_type& v)
                       { return std::cos(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            auto res = xsimd::sincos(input[i]);
            INFO("sincos / sin");
            CHECK_SCALAR_EQ(res.first, expected[i]);
            INFO("sincos / cos");
            CHECK_SCALAR_EQ(res.second, expected2[i]);
        }
    }

    void test_sinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sinh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::sinh(input[i]);
            INFO("sinh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_cosh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::cosh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::cosh(input[i]);
            INFO("cosh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_asin()
    {
        std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::asin(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::asin(ainput[i]);
            INFO("asin");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_acos()
    {
        std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::acos(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::acos(ainput[i]);
            INFO("acos");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_asinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::asinh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::asinh(input[i]);
            INFO("asinh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_acosh()
    {
        std::transform(acosh_input.cbegin(), acosh_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::acosh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::acosh(acosh_input[i]);
            INFO("acosh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_tan()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::tan(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::tan(input[i]);
            INFO("tan");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_cot()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return value_type(1.) / std::tan(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::cot(input[i]);
            INFO("cot");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_atan()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::atan(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::atan(input[i]);
            INFO("atan");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_acot()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return value_type(3.14159265358979323846 / 2.) - std::atan(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::acot(input[i]);
            INFO("acot");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }
};

template <class T>
struct scalar_complex_trigo_test
{
    using value_type = std::complex<T>;
    using real_value_type = T;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type expected;

    scalar_complex_trigo_test()
    {
        nb_input = 10000;
        input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(0.) + i * real_value_type(80.) / nb_input,
                                  real_value_type(0.1) + i * real_value_type(56.) / nb_input);
        }
        expected.resize(nb_input);
    }

    void test_sin()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sin(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::sin(input[i]);
            INFO("sin");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_cos()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::cos(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::cos(input[i]);
            INFO("cos");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_sincos()
    {
        vector_type expected2(nb_input);
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sin(v); });
        std::transform(input.cbegin(), input.cend(), expected2.begin(),
                       [](const value_type& v)
                       { return std::cos(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            auto res = xsimd::sincos(input[i]);
            INFO("sincos / sin");
            CHECK_SCALAR_EQ(res.first, expected[i]);
            INFO("sincos / cos");
            CHECK_SCALAR_EQ(res.second, expected2[i]);
        }
    }

    void test_sinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::sinh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::sinh(input[i]);
            INFO("sinh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_cosh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { return std::cosh(v); });
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::cosh(input[i]);
            INFO("cosh");
            CHECK_SCALAR_EQ(out, expected[i]);
        }
    }

    void test_asin()
    {
        using cbatch = xsimd::batch<value_type>;
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::asin(input[i]);
            cbatch bin(input[i]);
            value_type ref = xsimd::asin(bin).get(0);
            INFO("asin");
            CHECK(detail::bit_equal(out.real(), ref.real()));
            CHECK(detail::bit_equal(out.imag(), ref.imag()));
        }
    }

    void test_acos()
    {
        using cbatch = xsimd::batch<value_type>;
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::acos(input[i]);
            cbatch bin(input[i]);
            value_type ref = xsimd::acos(bin).get(0);
            INFO("acos");
            CHECK(detail::bit_equal(out.real(), ref.real()));
            CHECK(detail::bit_equal(out.imag(), ref.imag()));
        }
    }

    void test_asinh()
    {
        using cbatch = xsimd::batch<value_type>;
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::asinh(input[i]);
            cbatch bin(input[i]);
            value_type ref = xsimd::asinh(bin).get(0);
            INFO("asinh");
            CHECK(detail::bit_equal(out.real(), ref.real()));
            CHECK(detail::bit_equal(out.imag(), ref.imag()));
        }
    }

    void test_acosh()
    {
        using cbatch = xsimd::batch<value_type>;
        for (size_t i = 0; i < nb_input; ++i)
        {
            value_type out = xsimd::acosh(input[i]);
            cbatch bin(input[i]);
            value_type ref = xsimd::acosh(bin).get(0);
            INFO("acosh");
            CHECK(detail::bit_equal(out.real(), ref.real()));
            CHECK(detail::bit_equal(out.imag(), ref.imag()));
        }
    }
};

TEST_CASE_TEMPLATE("[scalar trigonometric | real]", T, float, double)
{
    scalar_real_trigo_test<T> Test;

    SUBCASE("sin")
    {
        Test.test_sin();
    }

    SUBCASE("cos")
    {
        Test.test_cos();
    }

    SUBCASE("sincos")
    {
        Test.test_sincos();
    }

    SUBCASE("sinh")
    {
        Test.test_sinh();
    }

    SUBCASE("cosh")
    {
        Test.test_cosh();
    }

    SUBCASE("asin")
    {
        Test.test_asin();
    }

    SUBCASE("acos")
    {
        Test.test_acos();
    }

    SUBCASE("asinh")
    {
        Test.test_asinh();
    }

    SUBCASE("acosh")
    {
        Test.test_acosh();
    }

    SUBCASE("tan")
    {
        Test.test_tan();
    }

    SUBCASE("cot")
    {
        Test.test_cot();
    }

    SUBCASE("atan")
    {
        Test.test_atan();
    }

    SUBCASE("acot")
    {
        Test.test_acot();
    }
}

TEST_CASE_TEMPLATE("[scalar trigonometric | complex]", T, float, double)
{
    scalar_complex_trigo_test<T> Test;

    SUBCASE("sin")
    {
        Test.test_sin();
    }

    SUBCASE("cos")
    {
        Test.test_cos();
    }

    SUBCASE("sincos")
    {
        Test.test_sincos();
    }

    SUBCASE("sinh")
    {
        Test.test_sinh();
    }

    SUBCASE("cosh")
    {
        Test.test_cosh();
    }

    SUBCASE("asin")
    {
        Test.test_asin();
    }

    SUBCASE("acos")
    {
        Test.test_acos();
    }

    SUBCASE("asinh")
    {
        Test.test_asinh();
    }

    SUBCASE("acosh")
    {
        Test.test_acosh();
    }
}
#endif
