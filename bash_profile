export LD_LIBRARY_PATH=/home/yokota/exafmm/wrapper:$LD_LIBRARY_PATH
export PATH=/home/yokota/gromacs/exec/bin:$PATH
export CMAKE_PREFIX_PATH=/home/yokota/exafmm/wrapper:$CMAKE_PREFIX_PATH
export CXX=g++
export CC=gcc
export CMAKE_C_FLAGS="-lstdc++ -ldl -lm"
export CMAKE_XX_FLAGS="-lstdc++ -ldl -lm"
export CMAKE_EXE_LINKER_FLAGS="-lstdc++ -ldl -lm"
#export OMP_NUM_THREADS=1