#!/bin/bash

set -e

mkdir -p build
cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
    -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
    -Dalpaka_DISABLE_VENDOR_RNG=ON \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_CUDA_ARCHITECTURES=52 \
    -G "Ninja" ..

ninja

./exe
