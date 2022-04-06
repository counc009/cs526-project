#include<stdio.h>
main()
{
int x, z;
z=0;
for(int y = 2; y<10; y++)
   {
     if (x%y == 0)
	  {
	   printf("%d is not a prime number",x);
	   z=1;
	   break;
      }
    }
if (z==0)
 {
  printf("%d is a prime number",x);
 }
}
