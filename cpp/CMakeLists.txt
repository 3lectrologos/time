cmake_minimum_required(VERSION 3.9)

project(diff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_VERBOSE_MAKEFILE on)

find_package(OpenMP REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(diff diff.cpp)
target_include_directories(diff PRIVATE eigen)
target_compile_options(diff PRIVATE -ffast-math)
target_include_directories(diff PRIVATE OpenMP_CXX_INCLUDE_DIRS)
target_link_libraries(diff PRIVATE OpenMP::OpenMP_CXX)

pybind11_add_module(tdiff tdiff.cpp)
target_include_directories(tdiff PRIVATE autodiff eigen)
target_include_directories(tdiff PRIVATE .)
target_compile_options(tdiff PRIVATE -ffast-math)
target_include_directories(tdiff PRIVATE OpenMP_CXX_INCLUDE_DIRS)
target_link_libraries(tdiff PRIVATE OpenMP::OpenMP_CXX)
