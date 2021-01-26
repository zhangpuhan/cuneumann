cmake_minimum_required(VERSION 3.19)
project(cuneumann LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cuneumann main.cpp model.cuh model.cu util.h kernel.cuh kernel.cu)

target_link_libraries(cuneumann -lcudart -lcusparse -lcublas -lcusolver)