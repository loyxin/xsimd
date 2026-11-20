############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         #
# Martin Renou                                                             #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 *
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. *
############################################################################

# doctest based unit tests for xsimd scalar math functions
# (include/xsimd/arch/xsimd_scalar_math.hpp). doctest is provided by the
# parent test/ directory (downloaded doctest.h or a found doctest package).
#
# Three independent test binaries are produced:
#  * test_xsimd_scalar_math  : basic scalar math entry points
#  * test_xsimd_scalar_perf  : performance comparison scalar vs libm
#  * test_xsimd_scalar_libm  : scalar implementation vs libm (float accuracy)
#  * test_xsimd_scalar_batch : scalar implementation vs batch, bit-exact,
#                              linked WITHOUT libm and verified as such.

# Each binary compiles the header standalone; DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
# is defined in the sources, so no separate implementation unit is needed.
function(xsimd_add_scalar_test tgt)
    add_executable(${tgt} ${ARGN})
    target_link_libraries(${tgt} PRIVATE xsimd)
    target_compile_features(${tgt} PRIVATE cxx_std_17)
    # The parent test/CMakeLists.txt already set up doctest for test_xsimd
    # (either the doctest::doctest imported target or the downloaded
    # doctest.h in the build tree); reuse exactly that setup.
    if (TARGET doctest::doctest)
        target_link_libraries(${tgt} PRIVATE doctest::doctest)
    else()
        target_include_directories(${tgt} PRIVATE "${CMAKE_BINARY_DIR}/test")
    endif()
    add_test(NAME ${tgt} COMMAND ${tgt})
endfunction()

# Performance knob: compile the scalar math kernels without NaN/Inf
# special-case handling. The batch bit-exactness checks are skipped in
# this configuration (see test_scalar_math_batch.cpp).
option(XSIMD_SCALAR_FAST_MATH "Skip NaN/Inf special-case handling in scalar math kernels (performance)" OFF)

# Binary 1: basic scalar math entry points (no libm comparison).
xsimd_add_scalar_test(test_xsimd_scalar_math test_scalar_math.cpp)

# Binary 2b: performance comparison scalar kernels vs libm.
xsimd_add_scalar_test(test_xsimd_scalar_perf test_scalar_perf.cpp)

# Binary 2: scalar implementation vs system libm (float accuracy).
xsimd_add_scalar_test(test_xsimd_scalar_libm test_scalar_math_libm.cpp)
if (XSIMD_SCALAR_FAST_MATH)
    target_compile_definitions(test_xsimd_scalar_libm PRIVATE XSIMD_SCALAR_FAST_MATH)
endif()

# Binary 3: scalar implementation vs batch, bit-exact, NO libm.
# To keep libm out of the runtime image entirely:
#  - doctest is a header-only library compiled straight into the binary
#    (no nextafter / ceilf references to satisfy: xsimd provides
#    libm-free scalar nextafter and ceil);
#  - libc++/libc++abi are linked statically so the shared libc++.so
#    (whose DT_NEEDED would pull in libm.so.6) is not used;
#  - static libgcc_eh (unwind) + compiler-rt builtins replace libgcc_s;
#  - -nostdlib++ -nodefaultlibs keep the driver from re-adding the
#    shared libraries after ours.
xsimd_add_scalar_test(test_xsimd_scalar_batch
    test_scalar_math_batch.cpp)
if (XSIMD_SCALAR_FAST_MATH)
    target_compile_definitions(test_xsimd_scalar_batch PRIVATE XSIMD_SCALAR_FAST_MATH)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_options(test_xsimd_scalar_batch PRIVATE
        -nostdlib++ -nodefaultlibs -no-pie)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # libc++ flavour
        target_link_libraries(test_xsimd_scalar_batch PRIVATE
            "-Wl,--start-group"
            -Wl,-Bstatic c++ c++abi
            "-Wl,--end-group"
            -Wl,-Bdynamic c pthread dl)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
            OUTPUT_VARIABLE _clang_resdir OUTPUT_STRIP_TRAILING_WHITESPACE)
        target_link_libraries(test_xsimd_scalar_batch PRIVATE
            "-Wl,--start-group" gcc_eh
            "${_clang_resdir}/lib/linux/libclang_rt.builtins-${CMAKE_SYSTEM_PROCESSOR}.a" c
            "-Wl,--end-group")
    else()
        # libstdc++ flavour (GCC): static libstdc++ + libgcc_eh; libgcc
        # provides the builtins, no compiler-rt needed.
        target_link_libraries(test_xsimd_scalar_batch PRIVATE
            "-Wl,--start-group"
            -Wl,-Bstatic stdc++ gcc_eh gcc
            "-Wl,--end-group"
            -Wl,-Bdynamic c pthread dl)
    endif()
endif()

# Verify the bit-exactness binary references no libm symbol.
add_custom_target(check_no_libm
    COMMAND ${CMAKE_COMMAND}
        -DMODEL=$<TARGET_FILE:test_xsimd_scalar_batch>
        -DNM=${CMAKE_NM}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoLibm.cmake
    DEPENDS test_xsimd_scalar_batch
    VERBATIM)
