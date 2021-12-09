cmake_minimum_required(VERSION 3.5)

project(kinodynamic-motion-planning-benchmark)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp REQUIRED)
pkg_check_modules(OMPL ompl REQUIRED)

# Enable C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# subdirectories
add_subdirectory(deps/fcl EXCLUDE_FROM_ALL)
add_subdirectory(deps/sbpl)# EXCLUDE_FROM_ALL)
# add_subdirectory(deps/ompl EXCLUDE_FROM_ALL)

# external projects
# include(ExternalProject)
# ExternalProject_Add(ompl
#   GIT_REPOSITORY    https://github.com/ompl/ompl.git
#   GIT_TAG           origin/1.5.2
# )

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${OMPL_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/deps/fcl/include
  ${CMAKE_SOURCE_DIR}/deps/fcl/include
  ${CMAKE_SOURCE_DIR}/deps/sbpl/src/include
  ${CMAKE_SOURCE_DIR}/deps/libMultiRobotPlanning/include
)

link_directories(
  ${OMPL_LIBRARY_DIRS}
)

add_library(motion_planning_common
  src/robotCarFirstOrder.cpp
  src/robotCarSecondOrder.cpp
)
set_property(TARGET motion_planning_common PROPERTY POSITION_INDEPENDENT_CODE ON)

# Applications

## ompl
add_executable(main_ompl
  src/main_ompl.cpp
  src/AO_RRT.cpp
)
target_link_libraries(main_ompl
  motion_planning_common
  ${Boost_LIBRARIES}
  ${OMPL_LIBRARIES}
  # ompl
  fcl
  yaml-cpp
)

## test_fcl
add_executable(test_fcl
  src/main_fcl.cpp
)
target_link_libraries(test_fcl
  ${Boost_LIBRARIES}
  # ${OMPL_LIBRARIES}
  fcl
  yaml-cpp
)

## main_sbpl
add_executable(main_sbpl
  src/main_sbpl.cpp
)
target_link_libraries(main_sbpl
  ${Boost_LIBRARIES}
  # ${OMPL_LIBRARIES}
  fcl
  yaml-cpp
  sbpl
)

## dbastar
add_executable(dbastar
  src/main_dbastar.cpp
)
target_link_libraries(dbastar
  motion_planning_common
  ${Boost_LIBRARIES}
  ${OMPL_LIBRARIES}
  fcl
  yaml-cpp
)

# Python bindings
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(motionplanningutils
  src/python_bindings.cpp
)
target_link_libraries(motionplanningutils PRIVATE
  motion_planning_common
  ${Boost_LIBRARIES}
  ${OMPL_LIBRARIES}
  fcl
  yaml-cpp
)