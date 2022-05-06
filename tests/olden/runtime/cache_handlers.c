/* $Log: cache_handlers.c,v $
 * Revision 1.4  1995/12/01  15:43:07  mcc
 * More up-to-date.  Some failed chip stuff.  Also have CWrite support for
 * smaller types (char, short)
 *
 * Revision 1.3  1995/10/06  14:39:54  mcc
 * Now can get sharers locally.  Also, don't core dump if invalidation of
 * non-existent line.
 *
 * Revision 1.2  1995/09/29  13:59:36  mcc
 * Added cache write capability for char, short.
 *
 *
 * handler routines for the Olden cache
 */

#include "constants.h"
#include "cache.h"

/* Want fewer false sharing messages */
#define SHAR_HASH(hash)  ((hash >> (PAGE_BITS)) & (SHAR_TBL_SZ-1)) 

struct sharers __SharTbl[SHAR_TBL_SZ];

#ifdef CACHE_STATS
int Handled = 0;
#endif

int __WriteCounter;

void __WriteDone()
{
  __WriteCounter++;
}

void __WaitForWrites(pn,num) {
  CMAML_poll_while_lt(&(__WritesReceived[pn]),num);
  CMAML_reply(pn,__WriteDone);
  __WritesReceived[pn] = 0;
}

void req1handler2(srcaddr,pn,rport)
int *srcaddr,pn,rport;
{
  int *realaddr = (int *) ((unsigned int) srcaddr >> (32-N_ADDR_BITS));
  unsigned int hash = (unsigned int) realaddr; 

  CACHE_STAT_INCR(Handled);
  
  CMAML_reply(pn, rep1handler2, *realaddr, *(realaddr+1), *(realaddr+2),
              *(realaddr+3));
  CMAML_reply(pn, rep2handler2, *(realaddr+4), *(realaddr+5), *(realaddr+6),
              *(realaddr+7));
  CMAML_reply(pn, rep3handler2, *(realaddr+8), *(realaddr+9), *(realaddr+10),
              *(realaddr+11));
  CMAML_reply(pn, rep4handler2, *(realaddr+12), *(realaddr+13), *(realaddr+14),
              *(realaddr+15));

  /* Here we want to record that we have a sharer for this line */
  hash = SHAR_HASH(hash);

  if (pn >= 32) {
    __SharTbl[hash].u.multiplex[1] |= (1 << (pn-32));
    }
  else {
    __SharTbl[hash].u.multiplex[0] |= (1 << pn);
    }
}

void rep1handler2(val1,val2,val3,val4)
int val1,val2,val3,val4;
{
  __InUse->values[0]=val1;
  __InUse->values[1]=val2;
  __InUse->values[2]=val3;
  __InUse->values[3]=val4;
  __done++;
}

void rep2handler2(val5,val6,val7,val8)
int val5,val6,val7,val8;
{
  __InUse->values[4]=val5;
  __InUse->values[5]=val6;
  __InUse->values[6]=val7;
  __InUse->values[7]=val8;
  __done++;
}

void rep3handler2(val1,val2,val3,val4)
int val1,val2,val3,val4;
{
  __InUse->values[8]=val1;
  __InUse->values[9]=val2;
  __InUse->values[10]=val3;
  __InUse->values[11]=val4;
  __done++;
}

void rep4handler2(val5,val6,val7,val8)
int val5,val6,val7,val8;
{
  __InUse->values[12]=val5;
  __InUse->values[13]=val6;
  __InUse->values[14]=val7;
  __InUse->values[15]=val8;
  __done++;
}

void writehandler(addr,val,pn,sz)
unsigned int *addr,val,pn,sz;
{
  switch(sz) {
  case 4: {
    unsigned int *realaddr = (unsigned int *) 
         ((unsigned int) addr >> (32-N_ADDR_BITS));
    *realaddr = val;
    break; 
    }
  case 2: {
    unsigned short *realaddr = (unsigned short *) 
         ((unsigned int) addr >> (32-N_ADDR_BITS));
    *realaddr = val;
    break; 
    }
  case 1: {
    unsigned char *realaddr = (unsigned char *) 
         ((unsigned int) addr >> (32-N_ADDR_BITS));
    *realaddr = val;
    break; 
    }
  default: 
    __PanicFlag = PANIC_WRITE_SZ;
    break;
  }
#ifdef VERIFY_WRITES
  CMAML_reply(pn,__WriteDone);
#endif
  __WritesReceived[pn]++;
}

void writehandlerD(addr,val1,val2,pn)
int *addr,val1,val2,pn;
{
  int *realaddr = (int *) ((unsigned int) addr >> (32-N_ADDR_BITS));
  *realaddr = val1;
  *(realaddr+1) = val2;
#ifdef VERIFY_WRITES
  CMAML_reply(pn,__WriteDone);
#endif
  __WritesReceived[pn]++;
}

void __SharerReply(mult0,mult1,ptr1,ptr2)
  int *ptr1, *ptr2, mult0, mult1;
{
  *ptr1 = mult0;
  *ptr2 = mult1;
  __done = 1;
}

void __LookupSharers(p,from_pn,ptr1,ptr2)
  int *ptr1,*ptr2,*p,from_pn;
{
  unsigned int hash = ((unsigned int) p >> (32-N_ADDR_BITS));
  
  hash = SHAR_HASH(hash);
  if (from_pn==__MyNodeId) {
    *ptr1 = __SharTbl[hash].u.multiplex[0];
    *ptr2 = __SharTbl[hash].u.multiplex[1];
  } 
  else {
    CMAML_reply(from_pn,__SharerReply,__SharTbl[hash].u.multiplex[0],
              __SharTbl[hash].u.multiplex[1],ptr1,ptr2);
  }
}

void __AcknowledgeInvalid()
{
  __InvalidationReplies++;
}

void __CInvalid2(p)
  unsigned int p;
{
  unsigned int tag = (p >> (PN_BITS+PAGE_BITS));
  unsigned int offset = ((p >> PN_BITS) & (PAGE_SIZE-1));
  unsigned int pn = (p & IDMASK);
  unsigned int hashval;
  unsigned int ourtag;
  unsigned int lineno = (offset >> LINE_BITS);
  unsigned int dirty = (1 << lineno);
  struct entry2 *entry;
 
  ourtag = (tag << PN_BITS) | pn;
  hashval = ((pn << 3) ^ tag) & (PG_TBL_SZ-1);
  for (entry = &(__PageTable[hashval]); 
       entry && entry->tag != ourtag; entry=entry->next) ;
  if (!entry) return;
  entry->valid = entry->valid & ~dirty;
}

void __InvalidateLine(p,from_pn)
  int *p, from_pn;
{
  __CInvalid2(p);
  CMAML_reply(from_pn,__AcknowledgeInvalid);
} 
