cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(gazebo REQUIRED)
find_library(YAMLCPP_LIBRARY  yaml-cpp)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# Model factory.
add_library(Factory SHARED src/Factory.cc)
target_link_libraries(Factory ${GAZEBO_LIBRARIES})

# HeisenPerson
add_library(HeisenPerson SHARED src/HeisenPerson.cc)
target_link_libraries(HeisenPerson ${GAZEBO_LIBRARIES})

# HeisenCamera
add_library(HeisenCamera SHARED src/HeisenCamera.cc)
target_link_libraries(HeisenCamera ${GAZEBO_LIBRARIES})

# Snapshots
add_library(Snapshot SHARED src/Snapshot.cc)
target_link_libraries(Snapshot ${GAZEBO_LIBRARIES} ${YAMLCPP_LIBRARY})
