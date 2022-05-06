# An LLVM Auto-parallelization via Decoupled Software Pipelining Pass

## Building the Pass
The CMake files provided in this reposity should work to build this pass out of
tree with LLVM 13.0.1.
To build, use the following commands from the directory containing this file.
```
mkdir build
cd build
EXPORT LLVM_DIR=<LLVM Location (see below)>
cmake ..
make
```
The LLVM Location should be the `lib/cmake/llvm` directory, if your LLVM build
is built using a `build/` directory this should be at
`<LLVM Root>/build/lib/cmake/llvm`.

## Running the Pass
This pass uses the Legacy Pass Manager, so to use the pass with `opt`, add the
flag `-enable-new-pm=0`.

To load the pass in `opt` add the `-load <path>/libPS-DSWP.so` where `<path>`
is the path to the `build/pass` directory in this repository (if you built
following the directions above).

The pass itself can be run with the `-psdswp` flag. There are several other
flags needed:
* `-num-threads=<t>` specifies that the pass should generate code using `<t>`
  threads for parallelized loops
* `-asll=<name>` specifies that the struct with name `<name>` is always used as
  an acyclic singly-linked list, and so our dependence analysis can assume this
  to be true; this option can be specified multiple times
  * Note: technically, in the LLVM IR it looks for a struct name `struct.<name>`,
    this is the naming convention of `clang` for translating C structs but may
    be different for other front-ends

The code that results must be linked with the `pwdswp.c` file found in the
`tests/` directory.

We also recommend running other LLVM optimization passes after, the code
we generate often contains repeated redundant casts that can be avoided.
Our `Makefile` for the test directory runs `-O3` optimizations.

## Building and Running Tests
In the `tests` directory, you can use the `Makefile` to build our tests.
Note that the `Makefile` assumes `clang` and `opt` can be located using your 
`PATH` environment variable.
The command `make all` builds all of our small test cases.
This commands will build both a `.ref` and a `.exec` executable, the `.ref` is
the reference compiled without our pass (but otherwise applying the same
optimizations) and the `.exec` is the parallelized version.

The Python script `runTests.py` in the `tests` directory can be used to repeatedly
run the test cases both normally and under `valgrind`.
It will run every test for which it finds a file matching the glob `*/*.exec`
and will compare output with the `.ref` version, reporting any differences.
