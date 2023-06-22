cmake_minimum_required(VERSION 3.22)
add_executable(vec vec.cpp)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(vec PRIVATE CUDA::cudart CUDA::nvrtc)
