# An LLVM Auto-parallelization via Decoupled Software Pipelining Pass
This pass uses the Legacy Pass Manager, so to use the pass with `opt`, add the
flag `-enable-new-pm=0`.

The `opt.sh` script can be used to run this pass (assuming LLVM 13.0.1 is
downloaded to `../llvm-13.0.1.src` and built under `../llvm-13.0.1.src/build`.
It also assumes `test.c` has been compiled to `test.ll` (`clang -S -emit-llvm
test.c`).
