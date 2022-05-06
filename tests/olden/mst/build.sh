#!/bin/bash
set -e

# We cannot build this since it tries to produce a struct type (and as it
# happens would be larger than 64-bits, so this really can't be handled)

clang -O1 -c -emit-llvm src/*.c
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o mst.exec
clang -O3 linked.opt.bc -lm -o mst.ref
rm *.bc
