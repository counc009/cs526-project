/* $Log: plain-mem.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

static int remaining = 0;
static char *temp;
#define ANDF 63

/* ANDF is used to force the data from malloc to be 4 byte aligned */
char *mymalloc(sz)
  int sz;
{
char *blah;

/*printf("malloc %d\n", sz);*/
 sz = (sz+ANDF) & ~ANDF;
if (sz>remaining) {
	temp=(char *)malloc(131072);
        if (!temp) chatting("Error! malloc returns null\n");
	blah=(char *) (((int) temp + ANDF) & ~ANDF);
   remaining = 131072 - (((int) blah) - ((int) temp));
   temp = blah;
}

blah=temp;
temp += sz;
remaining -= sz;

/*printf("returning 0x%x\n",blah);*/
return blah;
}

char *local_malloc(sz) 
  int sz;
{ return mymalloc(sz); }
