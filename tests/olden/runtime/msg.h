/* $Log: msg.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 */

#ifndef MSG_INCLUDED
#define MSG_INCLUDED

#define MAXTHREADS 200
#define NUM_MCBs 512

/* message types */
#define START_MSG 1
#define MIGRATE_ON_REF  2
#define MIGRATE_ON_REF_BUF  3
#define RPC_MIGRATE 4
#define RPC_REPLY 5
#define STOP_MSG  6
#define MIGRATE_ON_RET 7
#define MIGRATE_ON_RET0 7

#define MCB_NIL -1

#ifndef _ASM_

#ifdef MSG_STATS
extern int NumMigrations,NumRpcs;
#define MSG_STAT_INCR(x) x++
#else
#define MSG_STAT_INCR(x)
#endif

typedef struct mcb {
  void *address;
  int size;
  short from;
  short bufnum;
  int next;
} MCB;

extern int ThreadCount, __TransferDone, __QueueNotEmpty;
extern int __FreeMCB,__MigMCB,__RetMCB,__StopMCB,__MigBufMCB,__WakeMCB;
extern int __YourRport,BufferInUse;
extern MCB __MCBs[NUM_MCBs];
extern void __MsgDispatch();
extern void __update_frame();
extern void __clear_msgbuf();
extern void __recv_msg();
extern void __send_msg();
extern void __free_retbuf();
extern void IncBarrier();
extern void IsFinished();
extern void __TransmitDone();

#endif
#endif
