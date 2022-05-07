#!/bin/bash
rm *.bc
set -e

clang -O1 -c -emit-llvm src/*.c -DTORONTO
llvm-link *.bc -o linked.bc
opt -sccp -simplifycfg -loop-simplify linked.bc -o linked.opt.bc
opt -enable-new-pm=0 -load ../../../build/pass/libPS-DSWP.so -tbaa -psdswp -num-threads=8 linked.opt.bc -o linked.par.bc -stats
clang -O3 linked.par.bc ../../psdswp.ll -lm -o tsp.exec -lpthread
clang -O3 linked.opt.bc -lm -o tsp.ref -lpthread
rm *.bc
