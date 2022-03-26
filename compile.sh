#!/bin/bash

# I disable inlining because it makes understanding the LLVM IR and the results
# of our pass easier. We have -O1 to include some basic optimizations (including
# mem2reg
clang -S -emit-llvm -O1 -fno-inline test.c
../llvm-13.0.1.src/build/bin/opt -loop-simplify test.ll -o test.bc
../llvm-13.0.1.src/build/bin/llvm-dis test.bc -o test.ll
