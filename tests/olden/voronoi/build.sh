#!/bin/bash
set -e

clang -O1 -c -emit-llvm src/*.c
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 -asll=edge_rec linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o voronoi.exec
clang -O3 linked.opt.bc -lm -o voronoi.ref
rm *.bc
