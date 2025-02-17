# Force Release build type if not already specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optionally, add some default flags for Release mode for CUDA.
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# Similarly, if you have host code, add host compiler flags for Release.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Rest of your CMakeLists.txt...
cmake_minimum_required(VERSION 3.25)

project(cuda_hello LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories(${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(
    ${PROJECT_NAME}
    main.cu
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

link_directories(${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    cuda.lib
    ${OpenCV_LIBS} 
)
