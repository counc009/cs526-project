#!/bin/bash

# I disable inlining because it makes understanding the LLVM IR and the results
# of our pass easier. We have -O1 to include some basic optimizations (including
# mem2reg
clang -S -emit-llvm -O1 -fno-inline test.c
