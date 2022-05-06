#!/bin/bash
set -e

clang -O1 -c -emit-llvm src/*.c -DTORONTO
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 -asll=node_t linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o em3d.exec
clang -O3 linked.opt.bc -lm -o em3d.ref
rm *.bc
