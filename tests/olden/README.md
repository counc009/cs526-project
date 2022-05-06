In each benchmark there is a `build.sh` file to build a reference and
paralleized version.
Note that a few of these do not compile with clang, and a few do not compile
with our pass due to use of vector and struct typed virtual registers.
* bh: Appear to be type-issues and some undeclared variables, clang fails
* bisort: We successfully compile, but none of the loops are parallelizable
* em3d: We successfully compile, parallelizing 9 loops
* health: Our pass fails as it tries to produce/consume a vector
* mst: Our pass fails as it tries to produce/consume a struct
* perimeter: We successfully compile, but there are no loops in the IR
* power: Our pass fails as it tries to produce/consume a struct
* treeadd: We successfully compile, but there are no loops in the IR
* tsp: We successfully compile, having parallelized 3 loops
* voronoi: We successfully compile, having parallelized 6 loops
