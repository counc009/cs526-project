/* $Log: rpc.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 * RPC routines
 */

#include "constants.h"
#include "future-cell.h"
#include "msg.h"
#include "rpc.h"

extern void __RpcContinue();
extern void __RpcStolen();
extern void __MultiRpcResume();
extern void __OverflowRPC();
extern void __RpcReceive1();
extern void __RpcReceive2();
extern void __RpcReceive3();

int __CurRpcMsg, __RpcReplies, __FreeRpcReply, __RpcList, __FreeRpcMsg;
rpcmsg __RpcMsgs[NUM_RPCMSGs];
rpc_reply __RpcReplyMsgs[NUM_RpcReplies];

void __RpcFree(mcb)
     int mcb;
{
  CMAML_disable_interrupts();
  __RpcMsgs[mcb].next = __FreeRpcMsg;
  __FreeRpcMsg = mcb;
  CMAML_enable_interrupts();
}


void __RpcSend2(pn_to,rettype,argcount,pc,fcp,arg1,arg2,arg3,arg4,arg5,
                arg6,arg7)
     word_t pn_to,rettype,argcount,pc;
     future_cell_int *fcp;
     word_t arg1,arg2,arg3,arg4,arg5,arg6,arg7;
{
  word_t o2;

 
  /* Cache Coherence stuff */
  __MakeCoherent();

  if (pn_to >= __NumNodes) __Olden_panic("RPC to %d\n",pn_to);
  ThreadCount++;
  MSG_STAT_INCR(NumRpcs);
  
  if ((unsigned int) fcp & 3)
    /* Here we have a multirpc -- do right thing */
    {
      multi_cell_int *mcp = (multi_cell_int *) ((unsigned int) fcp - 2);

      mcp->sync->count++;
    }
  else 
    /* Have a normal future */
    {
      /* Caution for possible overflow */
      fcp->impl.next = (future_cell_int *) FC_EMPTY;
    }
  
  o2 = ((__MyNodeId << 16) | (argcount << 8) |
        rettype);
  __TransferDone = 0;
  CMAML_request(pn_to,__RpcReceive1,o2,pc,fcp,arg1);
  if (argcount > 1) {
    CMAML_disable_interrupts();
    CMAML_poll_while_lt(&__TransferDone,1);
    CMAML_enable_interrupts();
    CMAML_request(pn_to,__RpcReceive2,__CurRpcMsg,arg2,
                  arg3,arg4);
    if (argcount > 4) {
      CMAML_request(pn_to,__RpcReceive3,__CurRpcMsg,arg5,arg6,arg7);
    }
  }
  if ((unsigned int) fcp & 3)
    /* multirpc */
    {
      /* For now, have no overflow mechanism for MultiRpc's */
    }
  else
    /* normal future */
    {
      /*** Must overflow AFTER sending message ***/
      if (ThreadCount > MAXTHREADS) 
        {
          /*chatting("Overflowing RPC, to =%d,fcp=0x%x\n",pn_to,fcp);*/
          /*fcp->impl.next = (future_cell_int *) FC_EMPTY;*/
          /* see above */
          fcp->impl.returnPC = __GetRetAddr();
          NORETURN(0,fcp,__OverflowRPC);
        }
      fcp->impl.next = (future_cell_int *) FC_STOLEN;
    }
  
  /*CMAML_poll();*/
  /*CMAML_enable_interrupts();*/
  __GoExpress();
}

void __RpcReply(sp,fcp,rettype,ret1,ret2)
     word_t sp;
     future_cell_int *fcp;
     word_t rettype,ret1,ret2;
{
  ThreadCount--;

  /* lazy scheme IV */
  __ClearCache2();

  if ((unsigned int) fcp & 3)
    /* multicell */
    {
      multi_cell_int *mcp = (multi_cell_int *) ((unsigned int) fcp - 2);

      mcp->sync->count--;
      mcp->value = ret1;
      if ((rettype == 3) || (rettype > 4)) 
        {
          int *t = (((int *) mcp)+2);
          *t = ret1;
          *(t+1) = ret2;
        }
      if (mcp->sync->count == 0)
        if (mcp->sync->pc != 0) /* waiting */
          {
            CMAML_enable_interrupts();
            NORETURN(0,mcp->sync,__MultiRpcResume);
          }
      NORETURN(0,0,0,__MsgDispatch);
    }
  else
    /* regular rpc */
    {
      CMAML_enable_interrupts();
      fcp->value = ret1;
      if ((rettype == 3) || (rettype > 4))
        *(((int *) &(fcp->value))+1) = ret2;
      if (((int) fcp->impl.next & 3) == FC_EMPTY)
        {
          /*chatting("Continuing\n");*/
          NORETURN(0,fcp,__RpcContinue);
        }
      /*chatting("Other reply 0x%x, next is %d\n",fcp,fcp->impl.next);*/
      NORETURN(0,fcp,__RpcStolen);
    }
  
}

