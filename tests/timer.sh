
argN=1000
argSeed=314159

echo "Array"

printf "runningsum"
#time seq 100 | xargs -Iz ./array/runningsum.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/runningsum.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "incdec"
#time seq 100 | xargs -Iz ./array/incdec.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/incdec.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "squarenons"
#time seq 100 | xargs -Iz ./array/squarenons.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/squarenons.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecnext"
#time seq 100 | xargs -Iz ./array/incdecnext.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/incdecnext.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecprev"
#time seq 100 | xargs -Iz ./array/incdecprev.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/incdecprev.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "increment"
#time seq 100 | xargs -Iz ./array/increment.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/increment.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "sum"
#time seq 100 | xargs -Iz ./array/sum.ref $argN $argSeed > /dev/null 2>&1
time seq 100 | xargs -Iz ./array/sum.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

echo "Linked List"

printf "incdec"
time seq 100 | xargs -Iz ./linked_list/incdec.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "running sum"
time seq 100 | xargs -Iz ./linked_list/runningsum.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "squarenons"
time seq 100 | xargs -Iz ./linked_list/squarenons.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecnext"
time seq 100 | xargs -Iz ./linked_list/incdecnext.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecprev"
time seq 100 | xargs -Iz ./linked_list/incdecprev.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "increment"
time seq 100 | xargs -Iz ./linked_list/increment.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "sum"
time seq 100 | xargs -Iz ./linked_list/sum.exec $argN $argSeed > /dev/null 2>&1
echo "-----"


echo "Tree"

printf "incdec"
time seq 100 | xargs -Iz ./tree/incdec.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "squarenons"
time seq 100 | xargs -Iz ./tree/incdec.exec $argN $argSeed > /dev/null 2>&1
echo "-----"

printf "increment"
time seq 100 | xargs -Iz ./tree/incdec.exec $argN $argSeed > /dev/null 2>&1
echo "-----"


echo "Nested Array"

argN2=50
argM=50
argSeed=314159

printf "incdec"
time seq 100 | xargs -Iz ./nested_array/incdec.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "runningsum"
time seq 100 | xargs -Iz ./nested_array/runningsum.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "squarenons"
time seq 100 | xargs -Iz ./nested_array/squarenons.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecnext"
time seq 100 | xargs -Iz ./nested_array/incdecnext.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "incdecprev"
time seq 100 | xargs -Iz ./nested_array/incdecprev.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "increment"
time seq 100 | xargs -Iz ./nested_array/increment.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

printf "sum"
time seq 100 | xargs -Iz ./nested_array/sum.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

echo "Nested List" 
printf "increment"
time seq 100 | xargs -Iz ./nested_array/increment.exec $argN2 $argM $argSeed > /dev/null 2>&1
echo "-----"

