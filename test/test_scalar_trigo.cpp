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
    vector_type expected;

    scalar_real_trigo_test()
    {
        nb_input = 10000;
        input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(0.) + i * value_type(80.) / nb_input;
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
}
#endif
