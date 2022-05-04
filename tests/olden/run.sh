

#Usage: cd */src/
#Compiles all C files to bc files, links all bc files and than applies our pass

cd voronoi/src/
/usr/bin/clang-13 -c -emit-llvm -O1 -fno-inline *.c
~/compiler/llvm2/build/bin/llvm-link *.bc -o out.bc
~/compiler/llvm2/build/bin/opt -load ~/compiler/llvm2/build/lib/libPS-DSWP.so -enable-new-pm=0 -psdswp out.bc -o optout.bc -num-threads=8 -asll=node -debug

