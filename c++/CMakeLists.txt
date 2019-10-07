cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


project(DeepQC)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/usr/lib/libtorch/share/cmake/Torch") #Maybe you shold edit this line accordingly.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L /usr/local/lib -O3 -DNDEBUG  -fopenmp -lrt")



add_executable(DeepQC main.cpp)
target_link_libraries(DeepQC "${TORCH_LIBRARIES}")

