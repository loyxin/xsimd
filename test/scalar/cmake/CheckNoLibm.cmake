# Fails if the given binary (MODEL) references any dynamic libm symbol
# (sin, cos, exp, log, pow, scalbn, floor, ...).
#
# NM must be passed in on the command line (script mode has no cache):
#   cmake -DMODEL=<binary> -DNM=<nm-path> -P CheckNoLibm.cmake

if (NOT DEFINED NM)
    set(NM nm)
endif()

set(LIBM_SYMBOLS
    sin sinf cos cosf tan tanf asin acos atan
    sinh sinhf cosh coshf tanh tanhf
    exp expf log logf log2 log10 pow powf
    scalbn scalbnf floor floorf ceil ceilf
    nearbyint nearbyintf fma fmaf fmod hypot cbrt)

execute_process(
    COMMAND ${NM} -D --undefined-only ${MODEL}
    OUTPUT_VARIABLE NM_OUT
    RESULT_VARIABLE NM_RES)
if (NOT NM_RES EQUAL 0)
    message(FATAL_ERROR "nm failed on ${MODEL} (${NM_RES})")
endif()

set(FAILED 0)
foreach(sym IN LISTS LIBM_SYMBOLS)
    if (NM_OUT MATCHES "U[ \t]+${sym}(@|\n|$)")
        message(SEND_ERROR "test binary references libm symbol: ${sym}")
        set(FAILED 1)
    endif()
endforeach()

if (FAILED EQUAL 0)
    message(STATUS "check_no_libm: no libm dependency in ${MODEL}")
endif()
