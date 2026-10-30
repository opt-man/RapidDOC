cmake_minimum_required(VERSION 3.16.3)
project(figure8_mpc_ros1)

find_package(catkin REQUIRED COMPONENTS geometry_msgs roscpp)
catkin_package(CATKIN_DEPENDS geometry_msgs roscpp)

get_filename_component(F8_PACKAGE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
set(F8_NATIVE_ROOT "${F8_PACKAGE_SOURCE}/../training" CACHE PATH "Native training source")
set(F8_NATIVE_APPLICATION_BUILD "${F8_NATIVE_ROOT}/build" CACHE PATH
  "Build directory containing libf8_ours")
set(F8_NATIVE_LIBRARY_DIR "${F8_NATIVE_APPLICATION_BUILD}/bin")
find_path(F8_API_INCLUDE_DIR f8.h PATHS "${F8_NATIVE_ROOT}/include" NO_DEFAULT_PATH)
find_path(F8_MODEL_INCLUDE_DIR model_config.h PATHS "${F8_NATIVE_ROOT}/assets" NO_DEFAULT_PATH)
find_library(F8_OURS_LIBRARY NAMES f8_ours PATHS "${F8_NATIVE_LIBRARY_DIR}" NO_DEFAULT_PATH)
if(NOT F8_API_INCLUDE_DIR OR NOT F8_MODEL_INCLUDE_DIR OR NOT F8_OURS_LIBRARY)
  message(FATAL_ERROR "Build ../training with F8_BUILD_OURS=ON and set F8_NATIVE_APPLICATION_BUILD to its build directory")
endif()

add_executable(figure8_mpc_controller src/figure8_controller_node.cpp)
target_compile_features(figure8_mpc_controller PRIVATE cxx_std_17)
target_compile_options(figure8_mpc_controller PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(figure8_mpc_controller PRIVATE
  include ${catkin_INCLUDE_DIRS} "${F8_API_INCLUDE_DIR}" "${F8_MODEL_INCLUDE_DIR}")
target_link_libraries(figure8_mpc_controller "${F8_OURS_LIBRARY}" ${catkin_LIBRARIES})
set_target_properties(figure8_mpc_controller PROPERTIES
  BUILD_RPATH "${F8_NATIVE_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/..")

if(catkin_EXPORTED_TARGETS)
  add_dependencies(figure8_mpc_controller ${catkin_EXPORTED_TARGETS})
endif()

install(TARGETS figure8_mpc_controller RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
