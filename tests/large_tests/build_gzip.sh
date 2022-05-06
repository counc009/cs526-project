#!/bin/bash
set -e

# Building gzip fails because of a bug in LLVM's dependence analysis
# Similar bug report: https://github.com/llvm/llvm-project/issues/50612
clang -c -emit-llvm -O1 gzip.c -o gzip.bc
opt -sccp -simplifycfg -loop-simplify gzip.bc -o gzip.opt.bc
opt -load ../../build/pass/libPS-DSWP.so -enable-new-pm=0 -tbaa -psdswp -num-threads=8 gzip.opt.bc -o gzip.par.bc
clang -O3 gzip.par.bc ../psdswp.ll -o gzip.exec
clang -O3 gzip.opt.bc -o gzip.ref
