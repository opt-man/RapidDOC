cmake_minimum_required(VERSION 3.21)
project(RapidDOCImitation LANGUAGES CXX)

get_filename_component(RAPIDDOC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${RAPIDDOC_ROOT}/rapidoc" "${CMAKE_CURRENT_BINARY_DIR}/rapidoc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(fatrop 1.1.7 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(model "${CMAKE_CURRENT_SOURCE_DIR}/generated/imitation_models.c")
set_source_files_properties("${model}" PROPERTIES LANGUAGE CXX)
add_library(imitation_native SHARED imitation_native.cpp "${model}")
target_include_directories(imitation_native BEFORE PRIVATE
    "${FRS_FATROP_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/generated")
target_link_libraries(imitation_native PRIVATE factor_reuse_solver fatrop::fatrop Threads::Threads)
target_compile_definitions(imitation_native PRIVATE NOMINMAX FATROP_DISABLE_FINE_TIMINGS)
if(MSVC)
    target_compile_options(imitation_native PRIVATE
        /favor:INTEL64 /fp:precise $<$<CONFIG:Release>:/O2 /arch:AVX2>)
endif()
set_target_properties(imitation_native PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RAPIDDOC_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${RAPIDDOC_OUTPUT_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(imitation_native PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config} "${RAPIDDOC_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${config} "${RAPIDDOC_OUTPUT_DIR}")
endforeach()
