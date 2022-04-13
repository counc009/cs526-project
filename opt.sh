#!/bin/bash

../llvm-13.0.1.src/build/bin/opt -load build/pass/libPS-DSWP.so -enable-new-pm=0 -psdswp test.ll -o test.opt.bc -num-threads=8 -debug
