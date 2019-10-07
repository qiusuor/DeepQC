#  Requirement:

 *   Libtorch-1.2.0


We provide a compiled excutable binary file. IF it doesn't work, re-compile in following steps:

# Compile steps on Linux/Unix

*   Edit Torch_DIR in CMakeLists.txt accordingly
*   rm CMakeCache.txt
*   mkdir build && cd build
*   cmake ..
*   make
*   cp DeepQC ../

Make sure you have already installed CUDA, CUDNN, Libtorch successfully.

You may need to delete CMakeCache.txt when re-compile. Just ignore cmake warnings if appear.


# Have trouble with compiling?

Contact 750435412@qq.com. The response will be soon.
