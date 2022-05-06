/* $Log: threads.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 */

#ifndef THREAD_INCLUDED
#define THREAD_INCLUDED

/* from BlockedThread below */
#define BARRIER_FP 0
#define BARRIER_PC 4
#define BARRIER_NEXT 8
#define BARRIER_CHIP 12
#define BARRIER_SP 16

#ifndef _ASM_
typedef struct blkd_thd {
  word_t fp;
  word_t pc;
  struct blkd_thd *next;
  struct chip_list *chips;
  word_t sp;
} BlockedThread;

extern BlockedThread *Ready, *Waiting;
#endif


#endif
