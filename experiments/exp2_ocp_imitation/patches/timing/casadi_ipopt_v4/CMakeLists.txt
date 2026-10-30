cmake_minimum_required(VERSION 3.19)
project(casadi_ipopt_v4 LANGUAGES C CXX)
if(NOT WIN32 OR NOT MINGW)
  message(FATAL_ERROR "This private build currently requires x64 MinGW matching the Windows wheel ABI")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "An x64 compiler is required")
endif()
get_filename_component(FRS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(FRS_CASADI_PACKAGE "" CACHE PATH "Unmodified installed CasADi Python package")
set(FRS_CASADI_SOURCE "${FRS_ROOT}/.casadi-ipopt-v4/casadi-source" CACHE PATH "Matching CasADi source root")
set(_ipopt "${FRS_ROOT}/third_party/ipopt-highres")
set(_gen "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_gen}/casadi/interfaces/ipopt")
foreach(_path "${FRS_CASADI_PACKAGE}/libcasadi.dll.a"
    "${FRS_CASADI_PACKAGE}/libcoinmumps.dll.a"
    "${FRS_CASADI_PACKAGE}/libcasadi-tp-openblas.dll.a"
    "${FRS_CASADI_SOURCE}/casadi/interfaces/ipopt/ipopt_interface.cpp"
    "${_ipopt}/src/Common/IpSolveKernelTimer.hpp")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "Missing dependency: ${_path}")
  endif()
endforeach()
configure_file("${FRS_CASADI_PACKAGE}/include/coin-or/IpoptConfig.h"
               "${_gen}/config_ipopt.h" COPYONLY)
configure_file("${CMAKE_CURRENT_LIST_DIR}/config.h" "${_gen}/config.h" COPYONLY)
configure_file("${CMAKE_CURRENT_LIST_DIR}/casadi_nlpsol_ipopt_export.h"
               "${_gen}/casadi/interfaces/ipopt/casadi_nlpsol_ipopt_export.h" COPYONLY)

file(READ "${_ipopt}/src/Makefile.am" _makefile)
string(REPLACE "\r\n" "\n" _makefile "${_makefile}")
string(ASCII 92 _backslash)
string(REPLACE "${_backslash}\n" " " _makefile "${_makefile}")
string(REGEX MATCH "(^|\n)libipopt_la_SOURCES[ \t]*=[^\n]*" _base "${_makefile}")
string(REGEX MATCHALL "[A-Za-z0-9_/]+[.](cpp|c)" _paths "${_base}")
set(_sources "")
foreach(_path IN LISTS _paths)
  list(APPEND _sources "${_ipopt}/src/${_path}")
endforeach()
list(LENGTH _sources _source_count)
if(_source_count LESS 90)
  message(FATAL_ERROR "Failed to parse the upstream Ipopt source list")
endif()
foreach(_solver IpMc19TSymScalingMethod IpMa27TSolverInterface IpMa57TSolverInterface
    IpMa77SolverInterface IpMa86SolverInterface IpMa97SolverInterface
    IpPardisoSolverInterface IpMumpsSolverInterface)
  list(APPEND _sources "${_ipopt}/src/Algorithm/LinearSolvers/${_solver}.cpp")
endforeach()
set(_ipopt_includes "")
foreach(_part Common LinAlg LinAlg/TMatrices Algorithm Algorithm/LinearSolvers
    Algorithm/Inexact Interfaces contrib/CGPenalty)
  list(APPEND _ipopt_includes "${_ipopt}/src/${_part}")
endforeach()
add_library(frs_casadi_ipopt_v4 SHARED ${_sources} "${FRS_ROOT}/cmake/sipopt_timing_clock.cpp")
target_include_directories(frs_casadi_ipopt_v4 PRIVATE "${_gen}" ${_ipopt_includes}
  "${FRS_CASADI_PACKAGE}/include/coin-or/mumps")
target_compile_features(frs_casadi_ipopt_v4 PRIVATE cxx_std_11)
target_compile_definitions(frs_casadi_ipopt_v4 PRIVATE HAVE_CONFIG_H IPOPTLIB_BUILD DLL_EXPORT
  IPOPT_HAS_MUMPS=1 IPOPT_CHECKLEVEL=0 IPOPT_VERBOSITY=0 COIN_USE_MUMPS_MPI_H=1 NOMINMAX)
target_compile_options(frs_casadi_ipopt_v4 PRIVATE -O2 -fno-fast-math)
target_link_libraries(frs_casadi_ipopt_v4 PRIVATE
  "${FRS_CASADI_PACKAGE}/libcoinmumps.dll.a"
  "${FRS_CASADI_PACKAGE}/libcasadi-tp-openblas.dll.a")
set_target_properties(frs_casadi_ipopt_v4 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# Only the plugin registration name is changed. The complete upstream IPOPT
# interface implementation and its nominal/derivative mathematics are reused.
set(_interface_source "${FRS_CASADI_SOURCE}/casadi/interfaces/ipopt")
foreach(_name ipopt_interface.cpp ipopt_interface.hpp ipopt_interface_meta.cpp ipopt_nlp.cpp ipopt_nlp.hpp)
  file(READ "${_interface_source}/${_name}" _text)
  string(REPLACE "casadi_register_nlpsol_ipopt(" "casadi_register_nlpsol_ipopt_frs_v4(" _text "${_text}")
  string(REPLACE "casadi_load_nlpsol_ipopt(" "casadi_load_nlpsol_ipopt_frs_v4(" _text "${_text}")
  string(REPLACE "registerPlugin(casadi_register_nlpsol_ipopt)" "registerPlugin(casadi_register_nlpsol_ipopt_frs_v4)" _text "${_text}")
  string(REPLACE "plugin->name = \"ipopt\"" "plugin->name = \"ipopt_frs_v4\"" _text "${_text}")
  string(REPLACE "return \"ipopt\";" "return \"ipopt_frs_v4\";" _text "${_text}")
  string(REPLACE "#include \"../../core/" "#include \"casadi/core/" _text "${_text}")
  file(WRITE "${_gen}/casadi/interfaces/ipopt/${_name}" "${_text}")
endforeach()
file(READ "${_interface_source}/ipopt_runtime.hpp" _runtime)
file(WRITE "${_gen}/ipopt_runtime_str.h" "static const char* ipopt_runtime_str = R\"FRS_RUNTIME(${_runtime})FRS_RUNTIME\";\n")
add_library(casadi_nlpsol_ipopt_frs_v4 SHARED
  "${_gen}/casadi/interfaces/ipopt/ipopt_interface.cpp"
  "${_gen}/casadi/interfaces/ipopt/ipopt_interface_meta.cpp"
  "${_gen}/casadi/interfaces/ipopt/ipopt_nlp.cpp")
target_include_directories(casadi_nlpsol_ipopt_frs_v4 PRIVATE
  "${_gen}" "${FRS_CASADI_PACKAGE}/include" "${FRS_CASADI_SOURCE}"
  ${_ipopt_includes} "${FRS_CASADI_PACKAGE}/include/coin-or")
target_compile_features(casadi_nlpsol_ipopt_frs_v4 PRIVATE cxx_std_11)
target_compile_options(casadi_nlpsol_ipopt_frs_v4 PRIVATE -O2 -fno-fast-math -pthread)
target_compile_definitions(casadi_nlpsol_ipopt_frs_v4 PRIVATE HAVE_CONFIG_H
  CASADI_VERSION=31 CASADI_WITH_THREADSAFE_SYMBOLICS CASADI_WITH_THREAD
  CASADI_NLPSOL_IPOPT_BUILD NOMINMAX)
target_link_libraries(casadi_nlpsol_ipopt_frs_v4 PRIVATE frs_casadi_ipopt_v4
  "${FRS_CASADI_PACKAGE}/libcasadi.dll.a")
install(TARGETS frs_casadi_ipopt_v4 casadi_nlpsol_ipopt_frs_v4
  RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
