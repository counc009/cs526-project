#!/bin/sh


set -e 


echo "Compiling Perimeter"
cd perimeter
./build.sh
cd ..

echo "Compiling em3d"
cd em3d
./build.sh
cd ..

echo "Compiling bisort"
cd bisort
./build.sh
cd ..

echo "Compiling voronoi"
cd voronoi
./build.sh
cd ..

echo "Compiling treadd"
cd treeadd
./build.sh
cd ..

echo "Compiling tsp"
cd tsp
./build.sh
cd ..

