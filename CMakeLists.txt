cmake_minimum_required(VERSION 3.5)

project(kinodynamic-motion-planning-benchmark)

# find_package(Boost  REQUIRED COMPONENTS program_options)
# message(STATUS "Before croco")
# message(STATUS "${Boost_LIBRARIES}")
# find_package(crocoddyl REQUIRED)
# message(STATUS "after croco")
# message(STATUS "${Boost_LIBRARIES}")

find_package(crocoddyl REQUIRED)
find_package(Boost  REQUIRED COMPONENTS program_options unit_test_framework)

find_package(Eigen3 REQUIRED)
find_package(PkgConfig)








message(STATUS "Finding LZ4 libraries")
find_library(LZ4_LIBRARIES NAMES liblz4.so HINTS "lz4/lib")
if(LZ4_LIBRARIES)
    message(STATUS "Found: ${LZ4_LIBRARIES}")
else()
    message(STATUS "Not found: ${LZ4_LIBRARIES}")
    message(FATAL_ERROR "Cannot find required LZ4 libraries")
endif()

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
    ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/deps/sbpl/src/include
  ${CMAKE_SOURCE_DIR}/deps/libMultiRobotPlanning/include
  ${CMAKE_SOURCE_DIR}/deps/ompl/src
  ${CMAKE_BINARY_DIR}/deps/ompl/src
  ${CMAKE_SOURCE_DIR}/deps/rai/rai
)

add_library(motion_planning_common
  src/robots.cpp
  src/collision_checker.cpp
)

add_library(croco_ocp
  src/ocp.cpp
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

message(STATUS "${Boost_LIBRARIES}")

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

add_executable(evaluate_nn
  src/evaluate_nn.cpp
)

add_executable(test_croco
  src/test_croco.cpp 
)

add_executable(croco_main
  src/croco_main.cpp 
)



## TODO: add the graph as another file!


# set_source_files_properties(src/main_dbastar.cpp PROPERTIES COMPILE_OPTIONS "-g")


target_include_directories(dbastar
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)
target_link_libraries(dbastar
  motion_planning_common
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)


target_include_directories(croco_ocp PRIVATE ${CROCODDYL_INCLUDE_DIRS})
target_include_directories(test_croco PRIVATE ${CROCODDYL_INCLUDE_DIRS})
target_include_directories(croco_main PRIVATE ${CROCODDYL_INCLUDE_DIRS})



target_link_libraries(test_croco
  croco_ocp
  motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(croco_main
  croco_ocp
  motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)








target_include_directories(evaluate_nn
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)

target_link_libraries(evaluate_nn
  motion_planning_common
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
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
