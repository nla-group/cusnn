# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

add_executable(snn snn.cu example.cpp)

target_include_directories(snn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(snn PRIVATE CUDA::cudart CUDA::cublas)

# Set CUDA properties
set_target_properties(snn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80" # Adjust based on your GPU
)

# Enable debug symbols and define __DEBUG__
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(snn PRIVATE __DEBUG__)
    target_compile_options(snn PRIVATE -g) # -G for CUDA debug info
endif()

# Optimization flags for release build
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(snn PRIVATE -O3)
endif()
