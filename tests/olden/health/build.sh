#!/bin/bash
set -e

# Building this fails because we do not currently have support for producing
# vectors (it tries to produce a [2 x float] (which could be handled since it
# fits in a 64-bit quantity) but we simply lack support for it

clang -O1 -c -emit-llvm src/*.c
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o health.exec
clang -O3 linked.opt.bc -lm -o health.ref
rm *.bc
