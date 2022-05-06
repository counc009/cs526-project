/* $Log: debug.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

#include <stdio.h>
void chatting (fmt, a, b, c, d, e, f, g, h)
    char *fmt;
    int     a, b, c, d, e, f, g, h;
{
    char buf[256];

    sprintf(buf, "[%d] ", CMMD_self_address());
    sprintf(buf+strlen(buf), fmt, a, b, c, d, e, f, g, h);
    printf("%s", buf);
}

void dumpmemory(buf, size)
	int *buf;
	int size;
{
int x;
for (x=0; x<(size/4); x++) chatting("@[0x%x],0x%x\n",buf+x,*(buf+x));
}

void __Olden_panic(fmt,a,b,c,d,e,f,g,h)
     char *fmt;
     int a,b,c,d,e,f,g,h;
{
  char *buf[256];
  chatting("PANIC!!!\n");
  chatting(fmt, a, b, c, d, e, f, g, h);
  __ShutDown();
}



