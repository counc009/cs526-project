/* $Log: msg2.c,v $
 * Revision 1.4  1995/12/01  15:43:07  mcc
 * Support for eager futures.
 *
 * Revision 1.3  1995/10/06  14:40:31  mcc
 * modified cache stats slightly.
 *
 * Revision 1.2  1995/09/29  13:59:36  mcc
 * Now use #define'd PANICs.
 *
 * Revision 1.2  1995/09/19  15:12:36  mcc
 * Added cast to silence compiler warning.
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */
extern int Responded, DoBroadcast;
#ifdef EAGER
extern int Awaited_ec_done;
#endif
int __done,__value;
extern int BarrierAccum1, BarrierAccum2;

#include <cm/cmmd.h>
#include "msg.h"
#include "call-stub.h"
#include "future-cell.h"
#include "migrate.h"
#include "rpc.h"

#define assert(a) if (!(a)) chatting("MSG2: Assertion failed!\n")


void IncBarrier(mc,f,arg1,arg2)
     multi_cell_sync *mc;
     void (*f)();
     int arg1,arg2;
{
  Responded++;
  if (f) f(arg1,arg2,Responded);
  if (Responded==mc->count+1) 
    {
      Responded=0;
      DoBroadcast=1;
      __QueueNotEmpty++;
    }
}


void IsFinished(accum1,accum2)
{
  BarrierAccum1 = accum1;
  BarrierAccum2 = accum2;
  DoBroadcast = 1;
  __QueueNotEmpty++;
}

#ifdef CLEAR_STATS
void ClearStatsHandler()
{
#ifdef CACHE_STATS
extern int NumMisses, NumPageMisses;

  NumMisses = 0;
  NumPageMisses = 0;
#endif
#ifdef MSG_STATS
  NumRpcs = 0;
  NumMigrations = 0;
#endif
}
#endif


void increplhandler(value)
int value;
{
  __done = 1;
  __value = value;
}


void atomicinchandler(addr,from)
int *addr,from;
{
  int *realaddr = (int *) ((unsigned int) addr >> (32-N_ADDR_BITS));
  int val = *realaddr;

  *realaddr = val+1;
  CMAML_reply(from,increplhandler,val);
}

void __recv_msg(addr,size,type,from)
     int *addr,size,type,from;
{
  int mcb = __FreeMCB;

  if (mcb == MCB_NIL) 
    {
      __PanicFlag = PANIC_NO_MCB;
      __QueueNotEmpty = 1;
    }
  __FreeMCB = __MCBs[mcb].next;
  __MCBs[mcb].address = addr;
  __MCBs[mcb].from = (short) from;
  __MCBs[mcb].size = size;
  
  switch(type) 
    {
    case MIGRATE_ON_REF:
      __MCBs[mcb].next = __MigMCB;
      __MigMCB = mcb;
      break;
    case STOP_MSG:
      __MCBs[mcb].next = __StopMCB;
      __StopMCB = mcb;
#ifdef EAGER
      Awaited_ec_done = 3;
#endif
      break;
    case MIGRATE_ON_REF_BUF:
      __MCBs[mcb].next = __MigBufMCB;
      __MigBufMCB = mcb;
      break;
    default:
      if ((type >= MIGRATE_ON_RET0) && ((type-MIGRATE_ON_RET0) < NUMRETBUFS))
        {
          __MCBs[mcb].next = __RetMCB;
          __MCBs[mcb].bufnum = (short) (type-MIGRATE_ON_RET0);
          __RetMCB = mcb;
        }
      else {
          __PanicFlag = PANIC_MSG_TYPE;
          __QueueNotEmpty = 1;
        }
      break;
    }
  __QueueNotEmpty++;
}

void __TransmitDone()
{
  __TransferDone = 1;
}

void __send_msg(addr,sz,rport,to)
     void *addr;
     int sz,rport,to;
{
  /* 0 first arg means doubleword aligned, use 4 for word */
  /* 0 second argument means no offset */
  CMAML_scopy(to,rport,0,0,addr,sz,NULL,NULL);
}

void __clear_msgbuf()
{
  BufferInUse = 0;
}

void __setYourRport(rport)
  int rport;
{
  __YourRport = rport;
  __TransferDone = 1;
}

/* There are 128 rports */
static int __CodedSize[128];
static int __dest[128];
static void *__frame[128];

static void __PassItOn(rport)
  int rport;
{
  CMAML_request(__dest[rport],__recv_msg,__frame[rport],
              __CodedSize[rport],MIGRATE_ON_REF,__MyNodeId);
  CMAML_free_rport(rport);
}

void __update_frame(frame,codedsize,source,dest)
  void *frame;
  unsigned int codedsize,source;
{
  int rport = CMAML_allocate_rport();
  unsigned int size;

  /* extra -4 for hidden parameter */
  size = ((codedsize << 16) >> 16) -88 -4;
  __CodedSize[rport] = codedsize;
  __dest[rport] = dest;
  __frame[rport] = frame;
  CMAML_set_rport_handler(rport,__PassItOn);
  CMAML_set_rport_addr(rport,frame);
  CMAML_set_rport_byte_counter(rport,size);
  CMAML_reply(source,__setYourRport,rport);
}
  
void __free_retbuf(buf)
  int buf;
{
  __retmsgs[buf].retmsg.next_retmsg = __FreeRetMsg;
  __FreeRetMsg = buf;
}


