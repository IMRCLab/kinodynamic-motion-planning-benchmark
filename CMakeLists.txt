cmake_minimum_required(VERSION 3.5)

project(planning-with-interactions)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp)
pkg_check_modules(OMPL ompl)

# Enable C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include_directories(
  libMultiRobotPlanning/include
  ${EIGEN3_INCLUDE_DIR}
)

# Applications

## ompl_rrt
add_executable(ompl_rrt
  src/main_ompl_rrt.cpp
)
target_link_libraries(ompl_rrt
  ${Boost_LIBRARIES}
  ${OMPL_LIBRARIES}
  # yaml-cpp
)

## ompl_aorrt
add_executable(ompl_aorrt
  src/main_ompl_aorrt.cpp
)
target_link_libraries(ompl_aorrt
  ${Boost_LIBRARIES}
  ${OMPL_LIBRARIES}
  # yaml-cpp
)
