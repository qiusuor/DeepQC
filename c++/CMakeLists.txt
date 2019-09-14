cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


project(DeepQC)
set(CMAKE_CXX_STANDARD 14)
set(Torch_DIR "/home/qiusuo/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pipe  -L /usr/local/lib -Wall -O3 -DNDEBUG  -fopenmp -lrt")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -lrt -lprofiler")


add_executable(DeepQC main.cpp)
target_link_libraries(DeepQC "${TORCH_LIBRARIES}")





