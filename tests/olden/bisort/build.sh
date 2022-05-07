#!/bin/bash
rm *.bc
set -e

# Works, but does not actually parallelize any loops

clang -O1 -c -emit-llvm src/*.c
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o bisort.exec -lpthread
clang -O3 linked.opt.bc -lm -o bisort.ref -lpthread
rm *.bc
