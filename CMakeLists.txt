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

#target_link_libraries( opencv-setup ${OpenCV_LIBS})