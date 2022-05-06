/* $Log: mem.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

#define PN_BITS 7
#define LINE_SZ 64
#define ANDF 7

static int remaining = 0,local_remaining = 0;
static char *temp,*local_temp;

char *mymalloc(sz)
  int sz;
{
  extern unsigned int __MyNodeId;
  char *blah;
  unsigned int t1,t2;

  t1 = (unsigned int) temp;
  /* eight byte align me if necessary */
  if (sz>ANDF)
    {
      t2 = ((t1+ANDF) & ~ANDF);
      temp = (char *) t2;
      remaining -= (t2-t1);
      t1 = (unsigned int) temp;
    }

  /* If we can allocate all of this within the current cache line, and
     we have that much space remaining */
  if (((t1 & ~(LINE_SZ-1)) == ((t1+sz) & ~(LINE_SZ-1))) &&
      (remaining >= sz)) 
    {
      blah = temp;
      temp += sz;
      remaining -= sz;
      return ((char *) (((unsigned long) blah << PN_BITS) | __MyNodeId));
    }

  /* Start at beginning of next cache line */
  t2 = ((t1+(LINE_SZ-1)) & ~(LINE_SZ-1));
  remaining -= (t2-t1);
  temp = (char *) t2;
  
  if (sz>remaining) {
    temp=(char *)malloc(131072);
    if (!temp) __Olden_panic("Panic! malloc returns null\n");
    blah=(char *) (((unsigned int) temp + (LINE_SZ-1)) & ~(LINE_SZ-1));
    remaining = 131072 - (((unsigned int) blah) - ((unsigned int) temp));
    temp = blah;
  }
  
  blah=temp;
  temp += sz;
  remaining -= sz;
  
  return ((char *) (((unsigned long) blah << PN_BITS) | __MyNodeId));
}

char *local_malloc(sz)
  int sz;
{
  char *blah;
  unsigned int t1,t2;

  t1 = (unsigned int) local_temp;
  /* eight byte align me if necessary */
  if (sz>ANDF)
    {
      t2 = ((t1+ANDF) & ~ANDF);
      local_temp = (char *) t2;
      local_remaining -= (t2-t1);
    }
  
  if (sz>local_remaining) {
    local_temp=(char *)malloc(131072);
    if (!local_temp) __Olden_panic("Panic! malloc returns null\n");
    blah=(char *) (((unsigned int) local_temp + (LINE_SZ-1)) & ~(LINE_SZ-1));
    local_remaining = 131072 - (((unsigned int) blah) -
                                ((unsigned int) local_temp));
    local_temp = blah;
  }
  
  blah=local_temp;
  local_temp += sz;
  local_remaining -= sz;
  
  return blah;
}

