/* $Log: msg.c,v $
 * Revision 1.3  1995/12/01  15:43:07  mcc
 * Support for eager futures.
 *
 * Revision 1.2  1995/09/29  13:59:36  mcc
 * Added debugging info.  Also fixed bug with not using QueueNotEmpty for Ready.
 *
 * Revision 1.3  1995/09/21  18:57:22  mcc
 * Fixed bugs in poker chip alg.
 *
 * Revision 1.2  1995/09/19  15:11:23  mcc
 * Fixed several bugs in chip algorithm, most notably ReleaseAll.
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */
/* msg.c
 */
#include "migrate.h"
#include "msg.h"
#include "call-stub.h"
#include "future-cell.h"
#include "rpc.h"
#include "threads.h"
#include "cache.h"
#ifdef EAGER
#include "eager-cell.h"
#endif
#ifdef LAZY
#include "lazy.h"
#endif

int steal_target;

#ifdef MSG_DEBUG 
int __MsgDebug1 = -999;
int __MsgDebug2 = -999;
int __MsgDebug3 = -999;
#endif

static void TellYourFriends();

#define assert(a,b) if (!(a)) __Olden_panic("MSG: Assertion %d failed!\n",b)

#include <cm/cmmd.h>
#define roundup(x,n) (((x)+((n)-1))&(~((n)-1)))

int __MultiFutStk = 0;
aligned_retmsg __retmsgs[NUMRETBUFS];
static aligned_retmsg incomingretmsg;

/* For new synchronization */
BlockedThread *Ready = 0, *Waiting = 0;
int Responded = 0, DoBroadcast = 0;
int BarrierAccum1, BarrierAccum2;

int BufferInUse = 0;
int ThreadCount = 0;

#ifdef MSG_STATS
int NumMigrations = 0,NumRpcs = 0;
#endif

int __PanicFlag = 0;
MCB __MCBs[NUM_MCBs];
int __FreeMCB,__MigMCB,__RetMCB,__StopMCB,__QueueNotEmpty,__MigBufMCB;
int __WakeMCB;
int __OurRport,__YourRport;
int __TransferDone;

void __InitMsg() 
{
  int i;
  for (i=0; i<NUM_MCBs-1; i++) 
    {
      __MCBs[i].next = i+1;
    }
  for (i=0; i<NUMRETBUFS-1; i++)
    {
      __retmsgs[i].retmsg.next_retmsg = i+1;
    }
  for (i=0; i<NUM_RPCMSGs-1; i++)
    {
      __RpcMsgs[i].next = i+1;
    }
  for (i=0; i<NUM_RpcReplies-1; i++)
    {
      __RpcReplyMsgs[i].next = i+1;
    }
  __RpcReplyMsgs[NUM_RpcReplies-1].next = MCB_NIL;  
  __RpcMsgs[NUM_RPCMSGs-1].next = MCB_NIL;
  __retmsgs[NUMRETBUFS-1].retmsg.next_retmsg = MCB_NIL;
  __MCBs[NUM_MCBs-1].next = MCB_NIL;
  __FreeRpcReply = 0;
  __FreeRpcMsg = 0;
  __FreeMCB = 0;
  __FreeRetMsg = 0;
  __MigBufMCB = __RpcList = __RpcReplies = __MigMCB = __WakeMCB = 
    __RetMCB = __StopMCB = MCB_NIL;
  __QueueNotEmpty = 0;
  __OurRport = CMAML_allocate_rport();
  CMAML_set_rport_handler(__OurRport,__TransmitDone);
}

void __GetMessage(from,addr,sz,dest)
     int from,sz;
     void *addr,*dest;
{
  int nodeId = __MyNodeId;
  
  __TransferDone = 0;
  CMAML_set_rport_addr(__OurRport,dest);
  CMAML_set_rport_byte_counter(__OurRport,sz);
  if (from>=__NumNodes) __Olden_panic("Receive from %d?!?\n",from);
  CMAML_request(from,__send_msg,addr,sz,__OurRport,nodeId);
  CMAML_disable_interrupts();
  CMAML_poll_while_lt(&__TransferDone,1);
  CMAML_enable_interrupts();
}

  
/* __MsgDispatch: Allocate new work to this processor */
void __MsgDispatch(not_assert_fs_empty,fp,fnptr)
     int not_assert_fs_empty;
     char *fp, *fnptr;
{
  int from,size;
  future_cell_int *fc;
  void *addr;
  
  /* all calls to MsgDispatch either pass in a stack pointer
     (e.g. CallMsgSend), or a zero.  On zero, the future stack
     should be empty */
  if (!not_assert_fs_empty) assert(!__TopFutureStk(),0);


  /*chatting("Entering msg dispatch 0x%x\n",fp);*/
#ifndef EAGER
#ifdef LAZY
  if (!Ready) {
#endif
  /* Check for FUTUREs */
  fc = __PopFutureStk();
  if (fc!=0) 
    {
      /* Steal FUTURE or overflow as appropriate
         Recall placeholder for sp in NORETURNs */
          /* Overflow clears the future stack, storing it in
             the parent of the frame that just migrated.  The future
             stack is restored in __MigrateReturn.  We could at some
             point use Overflow to allow return messages to be
             processed first */
      if (ThreadCount > MAXTHREADS) 
        {
          NORETURN(0,fc,fp,__Overflow);
        }
      else 
        {
          __GoExpress();
          assert(fp,1);  assert(fnptr,2);
#ifdef VISUALIZE
          visFstackPop();
          visProcSteal();
#endif
#ifdef LAZY
          NORETURN(0,fc,fp,fnptr,__MyNodeId,__UnwindRemoteSteal);
#else
          NORETURN(0,fc,fp,fnptr,__Steal);
#endif
        }
    } 
#ifdef LAZY
  }
#endif
#else /* eager */
#ifdef EAGER
  if (TopEagerStack) {
    eager_steal_general *ec;
    pop_queue(TopEagerStack,ec);
    /*chatting("Stealing locally 0x%x, Top now 0x%x\n",ec,TopEagerStack);*/
    NORETURN(0,ec,__EagerStealLocal);
  }
#endif
#ifdef LAZY
#endif
#endif
  while(1) 
    {
      int mcb;

      if (__PanicFlag) __Olden_panic("Panic flag set %d\n",__PanicFlag);

      /* Are there any threads ready to go ? */
      if (Ready) 
        {
          BlockedThread *thread;
          
          __QueueNotEmpty--;
          thread = Ready;
#ifdef MSG_DEBUG
   __MsgDebug1 = (int) Ready;
   __MsgDebug2 = thread->pc;
#endif
          Ready = Ready->next;
          __SlowLane();
          
          NORETURN(0,thread,__BarrierResume);
        }

      if (DoBroadcast) 
        {
          __QueueNotEmpty--;
          Ready = Waiting;
          Waiting = 0;
          DoBroadcast = 0;

          TellYourFriends();

          if (Ready) 
            {
              BlockedThread *thread;
              
              thread = Ready;
              Ready = Ready->next;
              __SlowLane();
              NORETURN(0,thread,__BarrierResume);
            }
        }
      
      /* Check for RETURN messages */
      if (__RetMCB != MCB_NIL) 
        {
          int retbuf;
          
          /*chatting("About to receive return msg sp=0x%x\n",__getsp());*/
          /* Receive return message */
          CMAML_disable_interrupts();
          mcb = __RetMCB;
          from = (int) __MCBs[mcb].from;
          retbuf = (int) __MCBs[mcb].bufnum;
          addr = __MCBs[mcb].address;
          size = __MCBs[mcb].size;
          __RetMCB = __MCBs[mcb].next;
          __MCBs[mcb].next = __FreeMCB;
          __FreeMCB = mcb;
          __QueueNotEmpty--;
          CMAML_enable_interrupts();
          __GetMessage(from,addr,size,&(incomingretmsg.retmsg));
          CMAML_request(from,__free_retbuf,retbuf);
          __SlowLane();
          NORETURN(0,&(incomingretmsg.retmsg),0,__ReturnMsgRecv);
          /* placeholder 0 for sp in NORETURN */
        }
      if (__RpcReplies != MCB_NIL)
        {
          future_cell_int *fcp;
          word_t rettype,ret1,ret2;

          /*chatting("About to receive reply rpc,sp=0x%x\n",__getsp());*/
          CMAML_disable_interrupts();
          mcb = __RpcReplies;
          __RpcReplies = __RpcReplyMsgs[mcb].next;
          fcp = __RpcReplyMsgs[mcb].fcp;
          rettype = __RpcReplyMsgs[mcb].rettype;
          ret1 = __RpcReplyMsgs[mcb].ret[0];
          ret2 = __RpcReplyMsgs[mcb].ret[1];
          __RpcReplyMsgs[mcb].next = __FreeRpcReply;
          __FreeRpcReply = mcb;
          __QueueNotEmpty--;
          __SlowLane();
          NORETURN(0,fcp,rettype,ret1,ret2,__RpcReply);
          /* placeholder 0 for sp in NORETURN */
        }
#ifdef LAZY
      if (__WakeMCB != MCB_NIL) 
        {
          CMAML_disable_interrupts();
          mcb = __WakeMCB;
          __WakeMCB = __MCBs[mcb].next;
          CMAML_disable_interrupts();
          __ClearCache2();
          __ModifiedMask[0] = __ModifiedMask[1] = -1;
          __QueueNotEmpty--;
          __SlowLane();
          NORETURN(0,__MCBs[mcb].address,__lazy_continue);
        }
#endif
      /* Check for other types of messages */
      if (RPC_PRESENT)
        {
          POP_RPC(mcb);
          CLEAR_CACHE_MODIFIED;
          __ClearCache2();
          NORETURN(0,&(__RpcMsgs[mcb]),mcb,__RpcStub);
          /* placeholder 0 for sp in NORETURN */
        }          
      if (__MigMCB != MCB_NIL) 
        {
          /* receive migration message */
          CMAML_disable_interrupts();
          mcb = __MigMCB;
          __MigMCB = __MCBs[mcb].next;
          from = (int) __MCBs[mcb].from;
          addr = __MCBs[mcb].address;
          size = __MCBs[mcb].size;
          __MCBs[mcb].next = __FreeMCB;
          __FreeMCB = mcb;
          __QueueNotEmpty--;
          CMAML_enable_interrupts();
          NORETURN(0,from,addr,size,0,__CallMsgRecv);
        }
      if (__MigBufMCB != MCB_NIL) 
        {
          /*chatting("receiving buffered migration,sp=0x%x\n",__getsp());*/
          /* receive migration message */
          CMAML_disable_interrupts();
          mcb = __MigBufMCB;
          __MigBufMCB = __MCBs[mcb].next;
          from = (int) __MCBs[mcb].from;
          addr = __MCBs[mcb].address;
          size = __MCBs[mcb].size;
          __MCBs[mcb].next = __FreeMCB;
          __FreeMCB = mcb;
          __QueueNotEmpty--;
          CMAML_enable_interrupts();
          NORETURN(0,from,addr,size,1,__CallMsgRecv);
        }
      if (__StopMCB != MCB_NIL)
        {
          /* Don't need to receive stop message */
          Stats();
          exit(0);
        }

      if (__MultiFutStk) {
          multi_cell_sync *mcs = (multi_cell_sync *) __MultiFutStk;
          mcs->count++;
          __MultiFutStk = 0;
          __GoExpress();
          NORETURN(0,mcs,__MultiSteal);
          }

      /* Poll until a message arrives */
      /*chatting("Entering steal loop\n");*/
#ifdef EAGER
      POLL_EAGER(); /* send rejections to everyone */
#endif
#ifdef LAZY
      POLL_LAZY(); /* send rejections to everyone */
#endif
      CMAML_disable_interrupts();
#ifdef EAGER
      while (__QueueNotEmpty == 0) {
        /*int steal_target;*/
        extern int Awaited_ec_done;
        extern eager_steal_general Awaited_ec;
        
        steal_target = random() % (__NumNodes-1);
        if (steal_target >= __MyNodeId) steal_target++;

        Awaited_ec_done = 0;
        CMAML_request(steal_target,__StealRequest,__MyNodeId);
        CMAML_poll_while_lt(&Awaited_ec_done,1);
        if (Awaited_ec_done == 1) { /* steal succeeded */
          CMAML_enable_interrupts();
          /*chatting("Starting work 0x%x from %d\n",*/
                   /*Awaited_ec.impl.remote_ecp,Awaited_ec.impl.from_pe);*/
          NORETURN(0,&Awaited_ec,__EagerStealLocal);
        }
      }
#else
#ifdef LAZY
      {
        extern int InIdleLoop;
        InIdleLoop = 1;
        while (__QueueNotEmpty == 0) {
          /*int steal_target;*/
          
          steal_target = random() % (__NumNodes-1);
          if (steal_target >= __MyNodeId) steal_target++;
  
          CMAML_request(steal_target,__LazyStealRequest,__MyNodeId);
          CMAML_poll_while_lt(&__QueueNotEmpty,1);
          if (GotStealResponse < 0) { /* steal failed */
            __QueueNotEmpty += GotStealResponse;
            GotStealResponse = 0;
          }
        }
        if (GotStealResponse < 0) { /* steal failed */
            __QueueNotEmpty += GotStealResponse;
            GotStealResponse = 0;
        }
        InIdleLoop = 0;
      }
#else
      CMAML_poll_while_lt(&__QueueNotEmpty,1); 
#endif
#endif      
      /*CMAML_enable_interrupts(); commented out awhile ago*/

    }  /* while (1) */
} /* end __MsgDispatch */

   
int __AtomicInc(address)
     unsigned int address;
{
  unsigned int requst = address & IDMASK;
  __done = 0;

  if (requst >= __NumNodes) __Olden_panic("Inc on %d!?!\n",requst);
  CMAML_disable_interrupts();
  CMAML_request(requst,atomicinchandler,address,__MyNodeId);
  CMAML_poll_while_lt(&__done,1);
  CMAML_enable_interrupts();
  return __value;
}


/* remove me later if nothing added here */
void __ThreadSuspend2(bt,fnptr,fp)
  BlockedThread *bt;
  char *fnptr,*fp;
{
  NORETURN(1,fp,fnptr,__MsgDispatch);
}

void __Barrier2(bc,mc,f,arg1,arg2,fnptr,fp)
     BlockedThread *bc;
     char *mc, *fp, *fnptr,*f;
{
  CMAML_disable_interrupts();
  
  CMAML_request(0, IncBarrier, mc,f,arg1,arg2);

  /* Add me to the list of blocked threads */
  bc->next = Waiting;
  Waiting = bc;
  
  NORETURN(1,fp,fnptr,__MsgDispatch);
}

static void TellYourFriends()
{
  register int j=__MyNodeId;
  
  switch(j)
    {
    case 0:
      CMAML_request(16,IsFinished,BarrierAccum1,BarrierAccum2);
    case 16:
      CMAML_request(j+8,IsFinished,BarrierAccum1,BarrierAccum2);
    case 8:    case 24:
      CMAML_request(j+4,IsFinished,BarrierAccum1,BarrierAccum2);
    case 4: case 12: case 20: case 28:
      CMAML_request(j+2,IsFinished,BarrierAccum1,BarrierAccum2);
    case 2: case 6: case 10: case 14: case 18: case 22: case 26: case 30:
      CMAML_request(j+1,IsFinished,BarrierAccum1,BarrierAccum2);
      break;
    case 1:    case 3:    case 5:    case 7:    case 9:    case 11:
    case 13:    case 15:    case 17:    case 19:    case 21:    case 23:
    case 25:    case 27:    case 29:    case 31:
      break;
    }
}


