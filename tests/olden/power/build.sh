#!/bin/bash
set -e

# Can't build because it tries to produce/consume a struct

clang -O1 -c -emit-llvm src/*.c
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o power.exec
clang -O3 linked.opt.bc -lm -o power.ref
rm *.bc
