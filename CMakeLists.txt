cmake_minimum_required(VERSION 3.5)

# enable_testing()

project(kinodynamic-motion-planning-benchmark)
# include(ctest)

# find_package(Boost  REQUIRED COMPONENTS program_options)
# message(STATUS "Before croco")
# message(STATUS "${Boost_LIBRARIES}")
# find_package(crocoddyl REQUIRED)
# message(STATUS "after croco")
# message(STATUS "${Boost_LIBRARIES}")

find_package(crocoddyl REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Boost  REQUIRED COMPONENTS program_options unit_test_framework serialization)

find_package(Eigen3 REQUIRED)
find_package(PkgConfig)

# SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# SET(CMAKE_C_FLAGS_DEBUG "-g -O0")




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
# add_subdirectory(deps/rai EXCLUDE_FROM_ALL)

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
)

add_library(motion_planning_common
  src/robots.cpp
)

add_library(robot_models
  src/robot_models_base.cpp
  src/robot_models.cpp
  src/motions.cpp
)


add_library(croco_ocp
  src/ocp.cpp
  src/croco_models.cpp
)


add_library(dbastar
  src/dbastar.cpp
)


# add_library(ompl_geo
#   src/ompl_geo.cpp
# )
#




target_include_directories(motion_planning_common
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE deps/nigh/src 
)

set_property(TARGET motion_planning_common PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET robot_models PROPERTY POSITION_INDEPENDENT_CODE ON)

# Applications

## ompl
add_executable(main_ompl
  src/main_ompl.cpp
  src/ompl_sst.cpp
)

add_executable(main_check_robot
  src/main_check_robot.cpp
)

add_executable(main_idbastar
  src/main_idbastar.cpp
  src/idbastar.cpp
)


add_executable(test_idbastar
  unittest/test_idbastar.cpp
  src/idbastar.cpp
)






add_executable(test_ompl_sst
  unittest/test_ompl_sst.cpp
  src/ompl_sst.cpp
)

add_executable(dev_manifold
  src/solve_unicycle_R2SO2.cpp
)


# add_executable(example_nigh
#   src/example_nigh.cpp
# )



# target_include_directories(example_nigh PRIVATE deps/nigh/src )

target_include_directories(robot_models
  PRIVATE ${PINOCCHIO_INCLUDE_DIRS}
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)

target_include_directories(main_idbastar
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)

target_include_directories(test_idbastar
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  src
)



target_link_libraries(robot_models
  yaml-cpp
  fcl)

target_link_libraries(croco_ocp
    ${CROCODDYL_LIBRARIES} 
    robot_models
)




target_include_directories(main_ompl
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
)

target_include_directories(test_ompl_sst
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
  src
)



message(STATUS "${Boost_LIBRARIES}")

target_link_libraries(main_ompl
  motion_planning_common
  croco_ocp
  robot_models
  ${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(test_ompl_sst
  motion_planning_common
  robot_models
  croco_ocp
  ${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)



## main_ompl_geo
add_executable(main_ompl_geo
  src/main_ompl_geo.cpp
  src/ompl_geo.cpp
)
target_include_directories(main_ompl_geo
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
)



target_link_libraries(main_ompl_geo
  motion_planning_common
  robot_models
  croco_ocp 
  ${Boost_LIBRARIES}
  ${CROCODDYL_LIBRARIES} 
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


add_executable(main_dbastar
  src/main_dbastar.cpp
)

# add_executable(evaluate_nn
#   src/evaluate_nn.cpp
# )

add_executable(test_croco
  unittest/test_croco.cpp 
)

add_executable(test_db
  unittest/test_db.cpp 
)


add_executable(test_ompl_geo
  unittest/test_ompl_geo.cpp 
  src/ompl_geo.cpp
)




add_executable(croco_main
  src/croco_main.cpp 
)

add_executable(main_primitives
  src/main_primitives.cpp 
  src/generate_primitives.cpp 
)

add_executable(test_primitives
  unittest/test_primitives.cpp 
  src/generate_primitives.cpp 
)


## TODO: add the graph as another file!


# set_source_files_properties(src/main_dbastar.cpp PROPERTIES COMPILE_OPTIONS "-g")


target_include_directories(main_dbastar
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)

target_include_directories(test_db
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  src
)


target_include_directories(test_ompl_geo
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
PRIVATE ${CROCODDYL_INCLUDE_DIRS}
src
)








target_include_directories(dbastar
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
)

target_link_libraries(main_dbastar
  dbastar
  motion_planning_common
  robot_models
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(main_idbastar
  croco_ocp
  dbastar
  motion_planning_common
  robot_models
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(test_idbastar
  croco_ocp
  dbastar
  motion_planning_common
  robot_models
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)





target_link_libraries(test_db
  dbastar
  motion_planning_common
  robot_models
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(test_ompl_geo
  # ompl_geo
  motion_planning_common
  robot_models
  croco_ocp
  ${Boost_LIBRARIES}
  ${LZ4_LIBRARIES}
${CROCODDYL_INCLUDE_DIRS}
  ompl
  fcl
  yaml-cpp
)




target_include_directories(croco_ocp PRIVATE ${CROCODDYL_INCLUDE_DIRS})
target_include_directories(test_croco PRIVATE ${CROCODDYL_INCLUDE_DIRS} src)
target_include_directories(dev_manifold PRIVATE ${CROCODDYL_INCLUDE_DIRS} ${PINOCCHIO_INCLUDE_DIRS})



target_include_directories(croco_main PRIVATE ${CROCODDYL_INCLUDE_DIRS})
target_include_directories(main_primitives PRIVATE ${CROCODDYL_INCLUDE_DIRS})
target_include_directories(test_primitives PRIVATE ${CROCODDYL_INCLUDE_DIRS} src)



target_link_libraries(test_croco
  croco_ocp
  robot_models
  # motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(dev_manifold
  croco_ocp
  robot_models
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
)




target_link_libraries(croco_main
  croco_ocp
  robot_models
  # motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(main_primitives
  croco_ocp
  robot_models
  # motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(test_primitives
  croco_ocp
  robot_models
  # motion_planning_common
${CROCODDYL_LIBRARIES} 
${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)

target_link_libraries(main_check_robot
  robot_models
  # motion_planning_common
${Boost_LIBRARIES}
  fcl
  yaml-cpp
)





# target_include_directories(evaluate_nn
#   PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
#   PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
# )

# target_link_libraries(evaluate_nn
#   motion_planning_common
#   robot_models
#   ${Boost_LIBRARIES}
#   ${LZ4_LIBRARIES}
#   ompl
#   fcl
#   yaml-cpp
# )





# Python bindings
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(motionplanningutils
  src/python_bindings.cpp
)
target_include_directories(motionplanningutils
  PRIVATE ${CMAKE_BINARY_DIR}/deps/fcl/include
  PRIVATE ${CMAKE_SOURCE_DIR}/deps/fcl/include
  PRIVATE ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE deps/nigh/src
)

target_link_libraries(motionplanningutils PRIVATE
  motion_planning_common
  robot_models
  ${Boost_LIBRARIES}
  ompl
  fcl
  yaml-cpp
)






