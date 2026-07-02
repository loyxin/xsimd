cmake_minimum_required(VERSION 3.13)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xsimd-gtest)
    enable_testing()
    find_package(xsimd REQUIRED CONFIG)
endif()

# Resolve the xsimd target name. In-tree builds expose the xsimd::xsimd ALIAS,
# while an installed/exported xsimd only provides the plain `xsimd` target.
if(TARGET xsimd::xsimd)
    set(XSIMD_TARGET xsimd::xsimd)
elseif(TARGET xsimd)
    set(XSIMD_TARGET xsimd)
else()
    message(FATAL_ERROR "No xsimd target found.")
endif()


set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest)


function(xsimd_gtest_common target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE GTest::gtest_main ${XSIMD_TARGET})
endfunction()

# Helper: compile a source file as an OBJECT library with arch-specific flags.
#   xsimd_gtest_objlib(<target> <source> <arch_suffix> <cxx_flags>)
#   The resulting object library carries -DARCH_SUFFIX=<arch_suffix> so that
#   GTest test-suite names become e.g. ScalarVsVector_AVX2.
function(xsimd_gtest_objlib target source arch_suffix cxx_flags)
    add_library(${target} OBJECT ${source})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE GTest::gtest ${XSIMD_TARGET})
    target_compile_definitions(${target} PRIVATE ARCH_SUFFIX=${arch_suffix})
    separate_arguments(cxx_flags_list NATIVE_COMMAND "${cxx_flags}")
    target_compile_options(${target} PRIVATE ${cxx_flags_list})
endfunction()

# ---- Arch flags -----------------------------------------------------------
if(MSVC)
    set(_avx2_flags "/arch:AVX2")
    set(_avx512_flags "/arch:AVX512")
else()
    set(_avx2_flags "-mavx2;-mfma")
    set(_avx512_flags "-mavx512f")
endif()

xsimd_gtest_objlib(xsimd_sv_avx2   test_math_scalar_vs_vector.cpp AVX2   "${_avx2_flags}")
xsimd_gtest_objlib(xsimd_sv_avx512 test_math_scalar_vs_vector.cpp AVX512 "${_avx512_flags}")

# ---- Single test binary ---------------------------------------------------
add_executable(xsimd_gtest
    main.cpp
    test_math_std_vs_xsimd.cpp
    test_scalar_cmath.cpp
    $<TARGET_OBJECTS:xsimd_sv_avx2>
    $<TARGET_OBJECTS:xsimd_sv_avx512>
)
xsimd_gtest_common(xsimd_gtest)
