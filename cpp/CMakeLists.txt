cmake_minimum_required(VERSION 2.8)

project(diff)

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_VERBOSE_MAKEFILE on)

add_subdirectory(pybind11)

pybind11_add_module(diff diff.cpp)
target_include_directories(diff PRIVATE eigen)
target_compile_options(diff PRIVATE -fopenmp -ffast-math -O3 -std=c++17)
target_link_libraries(diff PRIVATE -fopenmp -ffast-math -O3)

pybind11_add_module(tdiff tdiff.cpp)
target_include_directories(tdiff PRIVATE autodiff eigen)
target_compile_options(tdiff PRIVATE -fopenmp -ffast-math -O3 -std=c++17)
target_link_libraries(tdiff PRIVATE -fopenmp -ffast-math -O3)
