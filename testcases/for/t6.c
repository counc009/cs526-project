#include<stdio.h>
main() 
{ 
int Max=999, Min=-1, Inpn=100, x; 
// Call the First number as current maximum and minimum
for(x=0;x<=5;++x) 
{ 
if(Inpn>Max) 
// if the next number is bigger than current maximum, store it 
{
Max = Inpn;
} 
if(Inpn<Min) 
// if the next number is lower than current minimum, store it 
{
Min = Inpn; 
} 
} 
printf(" The Maximum # is %d\n",Max);
printf(" The Minimum # is %d\n",Min);
}
