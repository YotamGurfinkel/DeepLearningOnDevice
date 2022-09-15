#!/usr/bin/env bash

git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make runtime -j4
