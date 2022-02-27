cmake_minimum_required(VERSION 3.5)

project(kinodynamic-motion-planning-benchmark)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp fcl REQUIRED)
# pkg_check_modules(OMPL ompl REQUIRED)

# Enable C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# subdirectories
add_subdirectory(deps/fcl EXCLUDE_FROM_ALL)
add_subdirectory(deps/sbpl)# EXCLUDE_FROM_ALL)
add_subdirectory(deps/ompl EXCLUDE_FROM_ALL)
add_subdirectory(deps/rai EXCLUDE_FROM_ALL)

# external projects
# include(ExternalProject)
# ExternalProject_Add(ompl
#   GIT_REPOSITORY    https://github.com/ompl/ompl.git
#   GIT_TAG           origin/1.5.2
# )

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/deps/sbpl/src/include
  ${CMAKE_SOURCE_DIR}/deps/libMultiRobotPlanning/include
  ${CMAKE_SOURCE_DIR}/deps/ompl/src
  ${CMAKE_BINARY_DIR}/deps/ompl/src
  ${CMAKE_SOURCE_DIR}/deps/rai/rai
)

add_library(motion_planning_common
  src/robots.cpp
)
target_include_directories(motion_planning_common
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
set_property(TARGET motion_planning_common PROPERTY POSITION_INDEPENDENT_CODE ON)

# Applications

## ompl
add_executable(main_ompl
  src/main_ompl.cpp
)
target_include_directories(main_ompl
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(main_ompl
  motion_planning_common
  ${Boost_LIBRARIES}
  # ${OMPL_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

## main_ompl_geometric
add_executable(main_ompl_geometric
  src/main_ompl_geometric.cpp
)
target_include_directories(main_ompl_geometric
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(main_ompl_geometric
  motion_planning_common
  ${Boost_LIBRARIES}
  # ${OMPL_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

## test_fcl
add_executable(test_fcl
  src/main_fcl.cpp
)
target_include_directories(test_fcl
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(test_fcl
  ${Boost_LIBRARIES}
  fcl
  yaml-cpp
)

## main_sbpl
add_executable(main_sbpl
  src/main_sbpl.cpp
)
target_include_directories(main_sbpl
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(main_sbpl
  ${Boost_LIBRARIES}
  fcl
  yaml-cpp
  sbpl
)

## dbastar
add_executable(dbastar
  src/main_dbastar.cpp
)
target_include_directories(dbastar
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(dbastar
  motion_planning_common
  ${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

## main_rai
add_executable(main_rai
  src/main_rai.cpp
  src/rai/car_utils.cpp
  src/rai/unicycle.cpp
  src/rai/trailer.cpp
  src/rai/quadrotor.cpp
  src/rai/solvers.cpp
)
target_link_libraries(main_rai
  rai
  yaml-cpp
)

## kinedit
add_executable(kinedit
  deps/rai/bin/src_kinEdit/main.cpp
)
target_link_libraries(kinedit
  rai
)

# Python bindings
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(motionplanningutils
  src/python_bindings.cpp
)
target_include_directories(motionplanningutils
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(motionplanningutils PRIVATE
  motion_planning_common
  ${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)
