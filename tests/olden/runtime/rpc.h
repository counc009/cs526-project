/* $Log: rpc.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 */

#ifndef RPC_INCLUDED
#define RPC_INCLUDED

#include "future-cell.h"

#define NUM_RPCMSGs 300
#define NUM_RpcReplies 200

#define ARGV 48
#define FCPLOC   56
#define FROMLOC 60
#define RETTYPELOC 52
#define SPLINKLOC 64
#define RPCSTUBSZ 128 /* 32 + 32 + 64 */
#define DBLRETLOC 8
#define STRSZ 12
#define RETLOC 16

#ifndef _ASM_
typedef struct
{
  unsigned char rettype;
  unsigned char argc;
  unsigned short from;
} rpcinfo;

typedef struct
{
  rpcinfo info;
  word_t PC;
  future_cell_int *fcp;
  word_t argv[7];
  int next;
} rpcmsg;


typedef struct
{
  word_t rettype;
  future_cell_int *fcp;
  word_t ret[3];
  int next;
} rpc_reply;

extern int __CurRpcMsg, __RpcReplies, __FreeRpcReply, __RpcList, __FreeRpcMsg;
extern rpcmsg __RpcMsgs[NUM_RPCMSGs];
extern rpc_reply __RpcReplyMsgs[NUM_RpcReplies];
extern void __RpcReply();
extern void __RpcStub();

#define RPC_NIL -1
#define RPC_PRESENT (__RpcList != RPC_NIL)
#define POP_RPC(mcb) {\
                 CMAML_disable_interrupts(); \
                 mcb = __RpcList; \
                 __RpcList = __RpcMsgs[mcb].next; \
                 __QueueNotEmpty--; \
                 CMAML_enable_interrupts(); \
                }
                 

#endif
#endif
