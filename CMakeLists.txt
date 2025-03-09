# Force Release build type if not already specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Minimum CMake version.
cmake_minimum_required(VERSION 3.25)

project(cuda_hello LANGUAGES CXX CUDA)

# Set CUDA standard.
set(CMAKE_CUDA_STANDARD 17)

# Use the native CUDA architecture.
set(CMAKE_CUDA_ARCHITECTURES native)

# Optionally add some default flags for Release mode for CUDA.
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math --expt-relaxed-constexpr -lineinfo -Xptxas -v")

# Similarly, add host compiler flags for Release.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find CUDA Toolkit and OpenCV.
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories.
include_directories(${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

# Create the executable.
add_executable(
    ${PROJECT_NAME}
    main.cu
)

# Enable separable compilation for CUDA.
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link directories if needed.
link_directories(${CUDAToolkit_LIBRARY_DIR})

# Link with CUDA and OpenCV libraries.
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    cuda.lib
    ${OpenCV_LIBS}
)
