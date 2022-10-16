cmake_minimum_required(VERSION 3.23)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_COMPILER /usr/bin/gcc-10)
set(GCC_COMPILE_FLAGS "-fopenmp -std=gnu++2a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}" )
add_executable(ParallelProgramming main.cpp)
