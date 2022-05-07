#!/bin/bash
rm *.bc
set -e

# We can build this, but there are no loops in the program to parallelize

clang -O1 -c -emit-llvm src/*.c -DTORONTO
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -o perimeter.exec -lpthread
clang -O3 linked.opt.bc -o perimeter.ref -lpthread
rm *.bc
