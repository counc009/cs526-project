/* $Log: cache.c,v $
 * Revision 1.3  1995/10/06  14:36:48  mcc
 * Added functionality for getting sharers.  Now use __WriteCounter, not
 * __done.
 *
 * Revision 1.2  1995/09/29  13:59:36  mcc
 * Added cache write capability for char, short.
 *
 *
 * Keeps the Olden cache
 */

#include "constants.h"
#include "cache.h"

extern int __WriteCounter;
unsigned int CACHE_DEBUG1=0, CACHE_DEBUG2=0;

unsigned int __ModifiedMask[2] = {0,0};
int __SomethingCached = 0;
struct CacheLine *__InUse;
struct entry2 __PageTable[PG_TBL_SZ];
int __WritesReceived[MAXPROC];
int __WritesSent[MAXPROC];
int __SentWrite = 0;
int __InvalidationReplies=0, __InvalidationCount=0;

#ifdef CACHE_STATS
int NumMisses,NumPageMisses;

void PrintCacheStats() 
{
  chatting("Misses: %d, PageMisses: %d\n",NumMisses,NumPageMisses);
  chatting("Invalidations Sent = %d\n", __InvalidationCount);
}

#endif

void __ClearSharers() 
{
  int i;

  for (i=0; i<SHAR_TBL_SZ; i++) {
    __SharTbl[i].u.multiplex[0] = 0;
    __SharTbl[i].u.multiplex[1] = 0;
  }
}

/* interrupts must be disabled */
void __GetSharerInfo(p,pi1,pi2)
  int *p,*pi1,*pi2;
{
  /* First get sharer information */
  if (PID(p)==__MyNodeId) {
    unsigned int hash = ((unsigned int) p >> (32-N_ADDR_BITS));   
  
    hash = (hash >> PAGE_BITS) & (SHAR_TBL_SZ-1);
    *pi1 = __SharTbl[hash].u.multiplex[0];
    *pi2 = __SharTbl[hash].u.multiplex[1];
    }
  else {
    __done = 0;
    CMAML_request(PID(p),__LookupSharers,p,__MyNodeId,pi1,pi2);
    /* be sure interrupts are disabled here */
    CMAML_poll_while_lt(&__done,1);
    }
}

/* interrupts must be disabled */
void __InvalidateSharers(p,mult0,mult1)
  int *p;
  int mult0, mult1;
{
  register int i,k;

  CACHE_DEBUG1 = (int) p;
  CACHE_DEBUG2 = -5;
  if (PID(p)!=__MyNodeId) __CInvalid2(p);

  /* Don't bother to send message to me! */
  if (__MyNodeId >= 32) mult1 &= ~(1 << (__MyNodeId - 32));
  else mult0 &= ~(1 << __MyNodeId); 
  
  /* Now tell everybody to invalidate these lines */
  
  for (i = 0, k=1; i<32; i++, k=k << 1) {
    if (mult0 & k) {
      CMAML_request(i,__InvalidateLine,p,__MyNodeId);
      __InvalidationCount++;
      }
    if (mult1 & k) {
      CMAML_request(i+32,__InvalidateLine,p,__MyNodeId);
      __InvalidationCount++;
      }
    }

  CMAML_poll_while_lt(&__InvalidationReplies,__InvalidationCount);
  CACHE_DEBUG2 = 0;
}

void __GetInfoInvalidateSharers(p)
  int *p;
{
  int a,b;
 
  CMAML_disable_interrupts();
  __GetSharerInfo(p,&a,&b);
  __InvalidateSharers(p,a,b);
  CMAML_enable_interrupts();
}
 
void __ClearCache2()
{
  /* What do we need to do to initialize the cache? */
  register int temp;

  __MakeCoherent();
  if (!__SomethingCached) return;
  
  for (temp=0; temp<PG_TBL_SZ; temp++) 
    {
      struct entry2 *ent;
      
      __PageTable[temp].valid = 0;
      ent = __PageTable[temp].next;
      while (ent) 
        {
          ent->valid = 0;
          ent = ent->next;
        }
    }
  __SomethingCached = 0;
}

void __InitCache2()
{
  /* What do we need to do to initialize the cache? */
  register int temp;

  for (temp=0; temp<PG_TBL_SZ; temp++) 
    {
      __PageTable[temp].tag = 0;
      __PageTable[temp].next = NULL;
    }
  for (temp=0; temp<MAXPROC; temp++)
    {
      __WritesSent[temp] = 0;
      __WritesReceived[temp] = 0;
    }
}

void __ClearCache2byMask(mask)
  unsigned int mask[2]; 
{
  register int temp;

  __MakeCoherent();
  for (temp=0; temp<PG_TBL_SZ; temp++)
    {
      struct entry2 *ent;
      
      for (ent = &(__PageTable[temp]); ent; ent = ent->next) 
        {
          unsigned int tag = ent->tag;
          /* recall tag of from [page # | proc_id ] */
          unsigned int proc_id = (tag & IDMASK);

          if (proc_id > 32) 
            {
              if ((1 << (proc_id-32)) & mask[1]) ent->valid = 0;
            }
          else
            {
              if ((1 << proc_id) & mask[0]) ent->valid = 0;
            }
        }
    }
}

/* recall address is of form:[14 bit page #|11 bit page offset|7 bit pn] */ 
int __PageMiss(address)
     unsigned int address;
{
  unsigned int proc_id = address & IDMASK;
  unsigned int hashval, tag;
  unsigned int *page,lineaddr,bitnum;
  struct entry2 *newpg_entry;
  int retval;
  unsigned int part2;
  
  CACHE_STAT_INCR(NumPageMisses);
  
  hashval = (proc_id << 3) ^ ((address >> 18) & (PG_TBL_SZ-1));
  tag = ((address >> 11) & ~IDMASK) | proc_id;

  page = (unsigned int *) local_malloc(PAGE_SIZE);
  if (__PageTable[hashval].tag == NULL) 
    {
      __PageTable[hashval].tag = tag;
      __PageTable[hashval].page_ptr = page;
      __PageTable[hashval].valid = 0;
      newpg_entry = &(__PageTable[hashval]);
    }
  else 
    {
      newpg_entry = (struct entry2 *) local_malloc(sizeof(*newpg_entry));
      newpg_entry->tag = tag;
      newpg_entry->page_ptr = page;
      newpg_entry->next = __PageTable[hashval].next;
      newpg_entry->valid = 0;
      __PageTable[hashval].next = newpg_entry;
    }
  bitnum = (address >> 13) & 31; /* 32 lines per page */
  lineaddr = (unsigned int) page;
  lineaddr = lineaddr + (bitnum << 6);
  /* page + cacheline_num * 64 */
  CACHE_DEBUG1 = __GetRetAddr();
  retval=__LineMiss(address,&(newpg_entry->valid),bitnum,lineaddr);

  return retval;
}

/* recall address is of form:[14 bit page #|11 bit page offset|7 bit pn] */ 
int __LineMiss(address,valid_vector,bitnum,lineaddr)
     unsigned int address;
     unsigned int *valid_vector;
     unsigned int bitnum;
     void *lineaddr;
{
  register int temp;
  unsigned int offset = (address & (63 << (32-N_ADDR_BITS)));
  register unsigned int vaddress = (unsigned int) (address^offset);
#ifdef VISUALIZE
  unsigned int pagenum = ((address & IDMASK) << 3) ^ 
                          ((address >> 18) & (PG_TBL_SZ-1));
#endif

  /* lazy scheme IV */
  __SomethingCached = 1;

  CACHE_STAT_INCR(NumMisses);
  
  __done = 0;
  __InUse = lineaddr;
  CACHE_DEBUG2 = __GetRetAddr();

#ifdef VISUALIZE
  visCacheMiss(pagenum);
#endif  
  CMAML_disable_interrupts();
  CMAML_request(address & IDMASK, req1handler2,vaddress,__MyNodeId);
  *valid_vector = (*valid_vector) | (1 << bitnum);
  CMAML_poll_while_lt(&__done,4);
  CMAML_enable_interrupts();

  return (((int) lineaddr) + (offset >> (32-N_ADDR_BITS)));
}

/* assumes 80 byte f.e. in pseudos.c */
void __CWrite2(address,value,sz)
unsigned int address,value,sz;
{
  unsigned int proc_id = (address & IDMASK);
  void *cache_addr;

  /* allocate on write */
  cache_addr = __AllocateOnWrite(address);

  /* Send request to remote processor */
#ifdef VERIFY_WRITES
   __WriteCounter = 0;
   CMAML_disable_interrupts();
#else
   __WritesSent[proc_id]++;
   __SentWrite = 1;
#endif
  CMAML_request(proc_id,writehandler,address,value,__MyNodeId,sz);

  /* lazy scheme IV */
  if (proc_id < 32)
    __ModifiedMask[0] = __ModifiedMask[0] | (1 << proc_id);
  else
    __ModifiedMask[1] = __ModifiedMask[1] | (1 << (proc_id-32));
    
  /* Update local cache copy */
  switch(sz) {
    case 4:
      *((unsigned int *) cache_addr) = value;
      break;
    case 2:
      *((unsigned short *) cache_addr) = value;
      break;
    case 1:
      *((unsigned char *) cache_addr) = value;
      break;
    default:
      __Olden_panic("Unrecognized size for CWrite, %d\n",sz);
    }
#ifdef VERIFY_WRITES
    CMAML_poll_while_lt(&__WriteCounter,1);
    CMAML_enable_interrupts();
#endif

}

/* assumes 80 byte f.e. in pseudos.c */
void __CWrite2D(address,value1,value2)
int address,value1,value2;
{
  unsigned int proc_id = (address & IDMASK);
  void *cache_addr;
  unsigned int *temp;

  cache_addr = __AllocateOnWrite(address);

  /* Send request to remote processor */
#ifdef VERIFY_WRITES
    __WriteCounter = 0;
    CMAML_disable_interrupts();
#else
    __WritesSent[proc_id]++;
    __SentWrite = 1;
#endif
  CMAML_request(proc_id,writehandlerD,address,value1,value2,__MyNodeId);

  /* lazy scheme IV */
  if (proc_id < 32)
    __ModifiedMask[0] = __ModifiedMask[0] | (1 << proc_id);
  else
    __ModifiedMask[1] = __ModifiedMask[1] | (1 << (proc_id-32));
    
  /* Update local cache copy */
  temp = (unsigned int *) cache_addr;
  *temp = value1;
  *(temp+1) = value2;
#ifdef VERIFY_WRITES
    CMAML_poll_while_lt(&__WriteCounter,1);
    CMAML_enable_interrupts();
#endif
}

void __MakeCoherent() 
{
  int temp;
  int to_do;

  if (!__SentWrite) return;
  CMAML_disable_interrupts();

  __WriteCounter = 0;
  to_do = 0;
  for (temp=0; temp < __NumNodes; temp++) {
    if (__WritesSent[temp]) {
      CMAML_request(temp,__WaitForWrites,__MyNodeId,__WritesSent[temp]);
      __WritesSent[temp] = 0;
      to_do++;
      }
    }
  __SentWrite = 0;
  CMAML_poll_while_lt(&__WriteCounter,to_do); 
  CMAML_enable_interrupts();
}
