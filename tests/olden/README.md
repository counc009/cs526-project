In each benchmark there is a `build.sh` file to build a reference and
paralleized version.

Here is a description of each of the benchmarks and our status for each of them. 
Note that a few of these do not compile with clang, and a few do not compile
with our pass due to use of vector and struct typed virtual registers.

| Name | Description |	Status |
| ------------- | ------------- | ------------- | 
| bh  | Barnes-Hut solves N-body problem using hierarchical methods | Appear to be type-issues and some undeclared variables, clang fails |
| bisort  | Sorts by creating two disjoint bitonic sequences and then merging them | We successfully compile, but none of the loops are parallelizable |
| em3d | Simulates propogation of electro-magnetic waves in 3D object | We successfully compile, parallelizing 9 loops |
| health | Simulates the Colombian health-care system | Our pass fails as it tries to produce/consume a vector |
| mst | Computes the minimum spanning tree of a graph | Our pass fails as it tries to produce/consume a struct |
| perimeter | Calculates the perimeter of a set of quad tree encoded raster images | We successfully compile, but there are no loops in the IR |
| power | Solves Power System Optimization problem | Our pass fails as it tries to produce/consume a struct |
| treeadd | Adds the values in a tree | We successfully compile, but there are no loops in the IR |
| tsp | Compues an estimate of the best hamiltonian circuit for Travelling salesman problem | We successfully compile, having parallelized 3 loops| 
| voronoi | computes Voronoi diagram of a set of points |  We successfully compile, having parallelized 6 loops |


If you are planning to build and execute, please ensure paths to CLANG, OPT and other object files are correctly provided to the build paths
