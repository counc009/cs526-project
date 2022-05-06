/* $Log: mem-ref.h,v $
 */

#include "constants.h"

extern void __Migrate(void *p);
extern void chatting(/* char *, ... */);
extern int SPMDInit(int argc, char *argv[]);
/* allocate "sz" bytes on processor "nd" */

extern char *mymalloc();
#define TO_PTR(proc) proc

#ifndef PLAIN

#include "future-cell.h"

extern void *LOCAL();
extern void *MLOCAL();
#define ALLOC(nd,sz)                   \
   (void *)((NONLOCAL(nd)) ?                                       \
       (MIGRATE((nd)), mymalloc(sz)) :              \
       mymalloc(sz))
#else
#define ALLOC(nd,sz) (char *)mymalloc(sz)
#define LOCAL(p) p
#define MLOCAL(p) p
#endif

