cmake_minimum_required(VERSION 3.10)

set(projectName 2dBlurFilter)
project(${projectName})

find_package(CUDA 12.2 REQUIRED)
find_package(OpenCV REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

include_directories(inc)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(
  ${projectName}
  src/filter.cu
  src/main.cpp
  inc/cuda_timer.h
  inc/filter.h
  inc/timer.h)

target_link_libraries(${projectName} ${OpenCV_LIBS})