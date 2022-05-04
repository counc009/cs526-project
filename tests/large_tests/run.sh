
/usr/bin/clang-13 -S -emit-llvm -O1 -fno-inline bzip2.c
~/compiler/llvm2/build/bin/opt -sccp -simplifycfg -loop-simplify  bzip2.ll -o bzip2.bc
~/compiler/llvm2/build/bin/llvm-dis bzip2.bc -o bzip2.ll
~/compiler/llvm2/build/bin/opt -load ~/compiler/llvm2/build/lib/libPS-DSWP.so -enable-new-pm=0  -tbaa -psdswp bzip2.ll -o bzip2.opt.bc -num-threads=8 -asll=node -debug


