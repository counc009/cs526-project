
/usr/bin/clang-13 -S -emit-llvm -O1 -fno-inline gzip.c
~/compiler/llvm13/build/bin/opt -loop-simplify gzip.ll -o gzip.bc
~/compiler/llvm13/build/bin/llvm-dis gzip.bc -o gzip.ll
~/compiler/llvm13/build/bin/opt -load ~/compiler/llvm13/build/lib/libPS-DSWP.so -enable-new-pm=0 -psdswp gzip.ll -o gzip.opt.bc -num-threads=8 -asll=node -debug

