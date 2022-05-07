
printf "voronoi"
time seq 10 | xargs -Iz ./voronoi/voronoi.ref> /dev/null 2>&1
time seq 10 | xargs -Iz ./voronoi/voronoi.exec> /dev/null 2>&1
echo "-----"


printf "em3d"
time seq 10 | xargs -Iz ./em3d/em3d.ref 2000 100 75 1 > /dev/null 2>&1
time seq 10 | xargs -Iz ./em3d/em3d.exec 2000 100 75 1 > /dev/null 2>&1
echo "---"

printf "bisort"
time seq 10 | xargs -Iz ./bisort/bisort.ref> /dev/null 2>&1
time seq 10 | xargs -Iz ./bisort/bisort.exec> /dev/null 2>&1
echo "---"

printf "treeadd"
time seq 10 | xargs -Iz ./treeadd/treeadd.ref> /dev/null 2>&1
time seq 10 | xargs -Iz ./treeadd/treeadd.exec> /dev/null 2>&1
echo "---"

printf "perimeter"
time seq 10 | xargs -Iz ./perimeter/perimeter.ref > /dev/null 2>&1
time seq 10 | xargs -Iz ./perimeter/perimeter.exec> /dev/null 2>&1
echo "---"

printf "tsp"
time seq 10 | xargs -Iz ./tsp/tsp.ref> /dev/null 2>&1
time seq 10 | xargs -Iz ./tsp/tsp.exec> /dev/null 2>&1
echo "---"




