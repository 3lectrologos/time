cmake_minimum_required(VERSION 3.9)

project(diff)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_VERBOSE_MAKEFILE on)

add_subdirectory(pybind11)

pybind11_add_module(diff diff.cpp)
target_include_directories(diff PRIVATE eigen)
target_compile_options(diff PRIVATE -ffast-math)
target_link_libraries(diff PRIVATE -ffast-math)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(diff PRIVATE OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(tdiff tdiff.cpp)
target_include_directories(tdiff PRIVATE autodiff eigen)
target_include_directories(tdiff PRIVATE .)
target_compile_options(tdiff PRIVATE -ffast-math)
target_link_libraries(tdiff PRIVATE -ffast-math)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(diff PRIVATE OpenMP::OpenMP_CXX)
endif()
