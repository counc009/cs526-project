
/usr/bin/clang-13 -S -emit-llvm -O1 -fno-inline gcc.c
~/compiler/llvm13/build/bin/opt -loop-simplify gcc.ll -o gcc.bc
~/compiler/llvm13/build/bin/llvm-dis gcc.bc -o gcc.ll
~/compiler/llvm13/build/bin/opt -load ~/compiler/llvm13/build/lib/libPS-DSWP.so -enable-new-pm=0 -psdswp gcc.ll -o gcc.opt.bc -num-threads=8 -asll=node -debug

