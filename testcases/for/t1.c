#include<stdio.h>
int main()
{
int sum; int x; sum=0;
 for(x=1;x<=50;++x) 
 // x take values in {1,2,3,...,50}
 {
 sum = sum + x; 
 }
 printf(" 1+2+...+50=%d\n",sum);
}
