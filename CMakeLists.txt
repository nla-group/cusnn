cmake_minimum_required(VERSION 3.18)

# Enable C++ and CUDA
project(CUSNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

file(GLOB SOURCES "src/*.cu" "example.cpp")

if(NOT SOURCES)
    message(FATAL_ERROR "No CUDA source files found in the src/ directory.")
endif()

add_executable(CUSNN ${SOURCES})

target_include_directories(CUSNN PRIVATE "./include")

target_link_libraries(CUSNN PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(CUSNN PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(CUSNN PRIVATE __DEBUG__)
    target_compile_options(CUSNN PRIVATE -g)
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(CUSNN PRIVATE -O3)
endif()
