/* $Log: rpc_handlers.c,v $
 * Revision 1.2  1995/09/29  13:59:36  mcc
 * Now use #define'd PANICs.
 *
 *
 */

#include "constants.h"
#include "msg.h"
#include "rpc.h"

void __SetRpcb();

void __RpcReceive1(rpc_info,pc,fcp,arg1)
     word_t rpc_info;
     word_t pc;
     future_cell_int *fcp;
     word_t arg1;
{
  int rpcb = __FreeRpcMsg;
  unsigned char rett,argc;
  unsigned short from;
  if (rpcb == RPC_NIL)
    {
      __PanicFlag = PANIC_NO_RPCB;
      __QueueNotEmpty = 1;
    }
  __FreeRpcMsg = __RpcMsgs[rpcb].next;
  from = rpc_info >> 16;
  argc = (rpc_info >> 8) & 255;
  rett = (rpc_info & 255);
  __RpcMsgs[rpcb].info.rettype = rett;
  __RpcMsgs[rpcb].info.argc = argc;
  __RpcMsgs[rpcb].info.from = from;
  __RpcMsgs[rpcb].argv[0] = arg1;
  __RpcMsgs[rpcb].PC = pc;
  __RpcMsgs[rpcb].fcp = fcp;
  if (argc == 1) 
    {
      __RpcMsgs[rpcb].next = __RpcList;
      __RpcList = rpcb;
      __QueueNotEmpty++;
    }
  else
    {
      if ((from < 0) || (from >= __NumNodes))
      {
        __PanicFlag = PANIC_RPC_SRC;
        __QueueNotEmpty = 1;
      }
      else
        CMAML_reply(from,__SetRpcb,rpcb);
    }
}

void __RpcReceive2(rpcb,arg2,arg3,arg4)
     word_t rpcb,arg2,arg3,arg4;
{
  unsigned char argc;

  __RpcMsgs[rpcb].argv[1] = arg2;
  __RpcMsgs[rpcb].argv[2] = arg3;
  __RpcMsgs[rpcb].argv[3] = arg4;
  argc = __RpcMsgs[rpcb].info.argc;
  if (argc <= 4) 
    {
      __RpcMsgs[rpcb].next = __RpcList;
      __RpcList = rpcb;
      __QueueNotEmpty++;
    }
}

void __RpcReceive3(rpcb,arg5,arg6,arg7)
     word_t rpcb,arg5,arg6,arg7;
{
  unsigned char argc;

  __RpcMsgs[rpcb].argv[4] = arg5;
  __RpcMsgs[rpcb].argv[5] = arg6;
  __RpcMsgs[rpcb].argv[6] = arg7;
  __RpcMsgs[rpcb].next = __RpcList;
  __RpcList = rpcb;
  __QueueNotEmpty++;
}

void __SetRpcb(rpcb)
     int rpcb;
{
  __CurRpcMsg = rpcb;
  __TransferDone = 1;
}

void __RpcReplyHandler(rettype,fcp,ret1,ret2)
     word_t rettype;
     future_cell_int *fcp;
     word_t ret1,ret2;
{
  int rpcb=__FreeRpcReply;

  if (rpcb == RPC_NIL)
    {
     __PanicFlag = PANIC_NO_RPC_REPB;
     __QueueNotEmpty = 1;
    }
  __FreeRpcReply = __RpcReplyMsgs[rpcb].next;
  __RpcReplyMsgs[rpcb].next = __RpcReplies;
  __RpcReplies = rpcb;
  __RpcReplyMsgs[rpcb].fcp = fcp;
  __RpcReplyMsgs[rpcb].rettype = rettype;
  __RpcReplyMsgs[rpcb].ret[0] = ret1;
  __RpcReplyMsgs[rpcb].ret[1] = ret2;
  if (rettype <= 8) __QueueNotEmpty++;
  else 
    {
      /* not yet implemented */
      __PanicFlag = PANIC_RPC_TYPE;
      __QueueNotEmpty = 1;
    }
}  
