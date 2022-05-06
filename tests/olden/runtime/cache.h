/* $Log: cache.h,v $
 * Revision 1.3  1995/12/01  15:47:19  mcc
 * added AllocateOnWrite.
 *
 * Revision 1.2  1995/10/06  14:38:45  mcc
 * Changed SHAR_TBL_SZ.
 *
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 * cache constants
 */

#ifndef CACHE_INCLUDED
#define CACHE_INCLUDED

#define NOVERIFY_WRITES
/* These are from declaration of struct entry2 below */
#define TAG_OFFSET 0 /* must be 0 or a bunch of things break */
#define PAGE_OFFSET 4
#define VALID_OFFSET 8
#define NEXT_OFFSET 12

#define PAGE_SIZE 2048 /* must be multiple of 2 */
#define PG_TBL_SZ 1024 /* number of entries in page table */
#define SHAR_TBL_SZ 4096
#define PAGE_BITS 11
#define LINE_BITS 6


#ifndef _ASM_
extern void *__AllocateOnWrite();
extern void __MakeCoherent();
extern void __SharerReply();
extern void __LookupSharers();
extern void __AcknowledgeInvalid();
extern void __InvalidateLine();
extern void __AcquireChip();
extern void __AcquireReply();
extern void __WaitForWrites();
extern void req1handler2();
extern void rep1handler2();
extern void rep2handler2();
extern void rep3handler2();
extern void rep4handler2();
extern void writehandler();
extern void writehandlerD();
extern void atomicinchandler();
extern void increplhandler();
extern void __WriteDone();

extern int __SentWrite;
extern int __WritesSent[MAXPROC],__WritesReceived[MAXPROC];
extern int __done,__value;
extern int __InvalidationReplies,__InvalidationCount;
extern unsigned int __ModifiedMask[2];

#define CLEAR_CACHE_MODIFIED {__ModifiedMask[0] = 0; __ModifiedMask[1] = 0;}

#ifdef CACHE_STATS
extern int NumMisses,NumPageMisses;
extern int Handled;
#define CACHE_STAT_INCR(x) (x)++
#else
#define CACHE_STAT_INCR(x)
#endif

/* make sure 8 byte aligned if using scopy */
struct CacheLine {
int values[16];
};

struct entry2 {
  unsigned int tag;
  void *page_ptr;
  unsigned int valid;
  struct entry2 *next;
};

struct sharers {
  union shar_union {
  unsigned char sharers[8];
  unsigned int multiplex[2];
  } u;
};

extern struct sharers __SharTbl[SHAR_TBL_SZ];
extern struct CacheLine *__InUse;
extern struct entry2 __PageTable[PG_TBL_SZ];
extern int __SomethingCached;

#endif /* ndef _ASM_ */

#endif /* CACHE_INCLUDED */










