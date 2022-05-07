#!/bin/bash
set -e

# The compilation fo bzip2 succeeds, but the resulting binary has errors , debugging
# this program is very difficult due to its size
clang -c -emit-llvm -O1 bzip2.c -o bzip2.bc
opt -sccp -simplifycfg -loop-simplify  bzip2.bc -o bzip2.opt.bc
opt -load ../../build/pass/libPS-DSWP.so -enable-new-pm=0 -tbaa -psdswp -num-threads=8 bzip2.opt.bc -o bzip2.par.bc
clang -O3 bzip2.par.bc ../psdswp.ll -o bzip2.exec -lpthread
clang -O3 bzip2.opt.bc -o bzip2.ref -lpthread
