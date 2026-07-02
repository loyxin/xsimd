/**
 * UT0: xsimd install smoke test
 *
 * Basic sanity checks that xsimd loads/stores and arithmetic work for the
 * default batch types (float / double). Linked against gtest_main, so no
 * manual main() is needed.
 */

#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
