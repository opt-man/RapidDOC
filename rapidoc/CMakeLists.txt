cmake_minimum_required(VERSION 3.21)
project(RapidDOC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FRS_FATROP_SOURCE_DIR "" CACHE PATH "Modified FATROP source tree")
get_filename_component(RAPIDDOC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(RAPIDDOC_OUTPUT_DIR "${RAPIDDOC_ROOT}/build/bin" CACHE PATH "Native library output")

if(NOT EXISTS "${FRS_FATROP_SOURCE_DIR}/src/ocp/aug_system_solver.cpp")
    message(FATAL_ERROR "Set FRS_FATROP_SOURCE_DIR to the FATROP source with the cache-export patch applied.")
endif()
find_package(fatrop 1.1.7 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
include(CheckIPOSupported)

add_library(factor_reuse_solver SHARED
    factor_reuse_solver.cpp
    "${FRS_FATROP_SOURCE_DIR}/src/ocp/dims.cpp"
    "${FRS_FATROP_SOURCE_DIR}/src/ocp/problem_info.cpp"
    "${FRS_FATROP_SOURCE_DIR}/src/ocp/jacobian.cpp"
    "${FRS_FATROP_SOURCE_DIR}/src/ocp/hessian.cpp"
    "${FRS_FATROP_SOURCE_DIR}/src/ocp/aug_system_solver.cpp"
    "${FRS_FATROP_SOURCE_DIR}/src/linear_algebra/lu_factorization.cpp")
target_include_directories(factor_reuse_solver BEFORE
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${FRS_FATROP_SOURCE_DIR}/include")
target_link_libraries(factor_reuse_solver PRIVATE
    fatrop::fatrop Threads::Threads OpenMP::OpenMP_CXX)
target_compile_definitions(factor_reuse_solver PRIVATE FRS_BUILD NOMINMAX FATROP_DISABLE_FINE_TIMINGS)
if(MSVC)
    target_compile_options(factor_reuse_solver PRIVATE
        /favor:INTEL64 /fp:precise $<$<CONFIG:Release>:/O2 /arch:AVX2>)
endif()
check_ipo_supported(RESULT RAPIDDOC_IPO_SUPPORTED)
if(RAPIDDOC_IPO_SUPPORTED)
    set_property(TARGET factor_reuse_solver PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
set_target_properties(factor_reuse_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RAPIDDOC_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${RAPIDDOC_OUTPUT_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(factor_reuse_solver PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config} "${RAPIDDOC_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${config} "${RAPIDDOC_OUTPUT_DIR}")
endforeach()
if(WIN32)
    add_custom_command(TARGET factor_reuse_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:factor_reuse_solver>
            $<TARGET_FILE_DIR:factor_reuse_solver>
        COMMAND_EXPAND_LISTS)
endif()
