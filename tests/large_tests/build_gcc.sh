#!/bin/bash
set -e

# Building GCC fails because of a bug in LLVM's dependence analysis
# Similar bug report: https://github.com/llvm/llvm-project/issues/50612
clang -c -emit-llvm -O1 gcc.c -o gcc.bc
opt -sccp -simplifycfg -loop-simplify gcc.bc -o gcc.opt.bc
opt -load ../../build/pass/libPS-DSWP.so -enable-new-pm=0 -tbaa -psdswp -num-threads=8 gcc.opt.bc -o gcc.par.bc
clang -O3 gcc.par.bc ../psdswp.ll -o gcc.exec -lpthread
clang -O3 gcc.opt.bc -o gcc.ref -lpthread
