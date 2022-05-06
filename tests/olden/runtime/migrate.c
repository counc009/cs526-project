/* $Log: migrate.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 * migration routines
 */

#include "constants.h"
#include "migrate.h"
#include "call-stub.h"
#include "cache.h"
#include "msg.h"

int __FreeRetMsg;
msg_t msg;
extern void __MigrateReturn();

/* __CallMsgSend: send migrate message to node that owns p
   If already migrated, try to copy message first, and cleanup,
   otherwise, update frame at original location, and lie about where
   this comes from */

void __CallMsgSend(p,frameSzB,argsize,bp,flag,migfp,fnptr)
     void *p;
     unsigned int frameSzB,argsize;
     refmsg_t *bp; /* also the stack pointer--see migrate.asm */
     void *migfp, *fnptr;
     int flag;
{
  unsigned int size;
  /* codedsize is [frameSzB,size] */
  unsigned int codedsize;
  unsigned int dest = (unsigned int) p & IDMASK;
  unsigned int source = __MyNodeId;
  refmsg_t *message_info;
  int *splink = (int *) bp - 2; /* used to clean up migrate frame */

  /*chatting("Sending 0x%x sz %d,%d to %d\n",migfp,frameSzB,argsize,dest);*/
  /* message-info moved above fp */
  message_info = (refmsg_t *) (((char *) bp) + frameSzB);

  /* Cache Coherence Stuff here */
  __MakeCoherent();

#ifdef VISUALIZE
  visProcMigOut();
#endif
  if (dest>=__NumNodes) __Olden_panic("Message to %d on pointer 0x%x?!?\n",
     dest,p);

  MSG_STAT_INCR(NumMigrations);
  size = frameSzB+argsize;
  codedsize = (frameSzB << 16) | size;
  if (flag) /* we did not previously migrate */ 
    {
      /* lazy scheme IV */
      message_info->modmask[0] = __ModifiedMask[0];
      message_info->modmask[1] = __ModifiedMask[1];

      /* Send message in usual fashion */
      CMAML_request(dest,__recv_msg,bp,codedsize,MIGRATE_ON_REF,source);
      ThreadCount++;
      *splink = (int) bp;
    }
  else 
    {
      int clean_size=size+CALLSTUB_SZ+RETMSG_SIZE;
      char *tmp = (char *) bp + clean_size;
      char *retmsgchar = (char *) bp + size + CALLSTUB_SZ;
      retmsg_t *retmsg = (retmsg_t *) retmsgchar;

      /* lazy scheme IV */
      message_info->modmask[0] = __ModifiedMask[0] | retmsg->modmask[0];
      message_info->modmask[1] = __ModifiedMask[1] | retmsg->modmask[1];

      if (message_info->nodeId == __MyNodeId) /* originally from us */
        {
          char *frame,*args,*args2;
          char *rwo,*frame_rwo;

          frame = (char *) message_info->savedFP;
          /* Don't copy rwo area */
          frame = frame - frameSzB;
          frame_rwo = frame + 64;
          rwo = ((char *) bp) + 64;
          memcpy(frame_rwo,rwo,frameSzB-64-12);
          /* Don't overwrite hidden parameter, stored at fp+64 */
          args = (char *) bp;
          args = args + frameSzB;
          args2 = ((char *) message_info->savedFP);
          memcpy(args2,args,64);
          memcpy((args2+68),(args+68),argsize-68);
          CMAML_request(dest,__recv_msg,frame,codedsize,MIGRATE_ON_REF,source);
        }
      else
        /* Send message by first copying, or updating original frame */
        if (BufferInUse) 
          {
            /* update original frame, then lie about origin */
            /* We send size-16 and avoid overwriting fc,o7,fp 
               by two calls to scopy */
            /* Note we can use 16 (not 12), b/c CSBUF_reserved -- 
               an unused location is at 16 */
            int newsource = message_info->nodeId;
            char *frame = (char *) message_info->savedFP;
            char *argarea = (char *) bp;

            argarea += frameSzB;
            frame = frame - message_info->frameSzB;
            /* somehow send this frame to newsource */
            if (newsource >= __NumNodes)
              __Olden_panic("newsource is %d?!?\n",newsource);
            __TransferDone = 0;
            CMAML_request(newsource,__update_frame,frame,codedsize,
                          source,dest);
            CMAML_disable_interrupts();
            CMAML_poll_while_lt(&__TransferDone,1);
            /* Can't call these with interrupts enabled */
            CMAML_scopy(newsource,__YourRport,0,72,((char *) bp)+72,
                        frameSzB-88,NULL,NULL);
            CMAML_scopy(newsource,__YourRport,0,frameSzB,argarea,
                        64,NULL,NULL);
            CMAML_scopy(newsource,__YourRport,0,frameSzB+68,(argarea+68),
                        argsize-68,NULL,NULL);
            CMAML_enable_interrupts();
            /* 0 doubleword, 0 offset, blocking since not in handler */
/*** Now using PassItOn strategy
            CMAML_request(dest,__recv_msg,frame,codedsize,
                          MIGRATE_ON_REF,newsource);
***/
          }
        else 
          {
            char *rwo,*msg_rwo;

            BufferInUse = 1;
            /* copy to msg, don't bother with rwo area  */
            rwo = ((char *) bp) + 64;
            msg_rwo = ((char *) &(msg.ref)) + 64;
            memcpy(msg_rwo,rwo,size-64);
            /*memcpy(&(msg.ref),bp,size);*/
            /* send from msg */
            CMAML_request(dest,__recv_msg,&(msg.ref),codedsize,
                          MIGRATE_ON_REF_BUF,source);
          }
      *splink = (int) tmp; /* clean up the migrated frame and
                              __Migrate's frame also */
    }
  NORETURN(1,migfp,fnptr,__MsgDispatch);
  /* look for more work, recall NORETURN puts sp in first argument */
}

/* Deliberately sets stack pointer-- frame from CallMsgRecv overlaps
   the return message, and handling this is done in CallStub
   The call to CallStub does not return :
   WARNING!  must set splink lest GetMessage do something bad,
   extra -8 added to sp, removed at passing to CallStub

   clear_buf means if this was a MIGRATE_ON_REF_BUF 

   codedsize is [frameSzB,size]
*/
void __CallMsgRecv(sp,from,addr,codedsize,clear_buf)
     char *sp,*addr;
     unsigned int from,codedsize,clear_buf;
{
  unsigned int size2,size;
  unsigned int frameSzB;
  refmsg_t *callMsg;
  retmsg_t *retMsg;
  char *getme;
  int *splink;

#ifdef VISUALIZE
  visProcMigIn();
#endif
  frameSzB = codedsize >> 16;
  size = (codedsize << 16) >> 16;
  size2 = size+64;
  sp -= (RETMSG_SIZE+size2+CALLSTUB_SZ+8);
  __setsp(sp);
  splink = (int *) (sp+64); /* must use int * to get full 0 stored */
  *splink = 0; /* spaghetti stack fix */
  /* bottom 64 bytes are reg. window -- do not bother to send */
  getme = (sp+72+64);
  callMsg = (refmsg_t *) (sp+72);
  __GetMessage(from,addr+64,size-64,getme);
  if (clear_buf) CMAML_request(from,__clear_msgbuf);

  /* lazy Scheme 4 */
  retMsg = (retmsg_t *) (sp+8+size2+CALLSTUB_SZ);
  if (__MyNodeId > 32) {
     retMsg->modmask[0] = __ModifiedMask[0];
     retMsg->modmask[1] = __ModifiedMask[1] | (1 << (__MyNodeId-32));
  }
  else {
     retMsg->modmask[0] = __ModifiedMask[0] | (1 << __MyNodeId);
     retMsg->modmask[1] = __ModifiedMask[1];
  }

  __ClearCache2();
  __ModifiedMask[0] = 0;
  __ModifiedMask[1] = 0;
  
  __CallStub(callMsg,sp+8,frameSzB,size2);
  /* first instruction sets sp to 2nd argument */
}

void __CallMsgReply(nodeId,retMsg)
     word_t nodeId;
     retmsg_t *retMsg;
{
  int size = sizeof(retmsg_t)-STRUCTSIZE+retMsg->retsize;
  int source = __MyNodeId;
  int retmsgnum;

  /*chatting("Returning 0x%x to %d\n",retMsg,nodeId);*/
  if (nodeId >= __NumNodes) __Olden_panic("Return to %d?!?\n",nodeId);

  /* lazy scheme IV */
  retMsg->modmask[0] = retMsg->modmask[0] | __ModifiedMask[0];
  retMsg->modmask[1] = retMsg->modmask[1] | __ModifiedMask[1];

  /* Cache coherence stuff */
  __MakeCoherent();

  /* Figure out a place to put this one */
  CMAML_disable_interrupts();
  retmsgnum = __FreeRetMsg;
  __FreeRetMsg=__retmsgs[retmsgnum].retmsg.next_retmsg;
  CMAML_enable_interrupts();
  if (retmsgnum == MCB_NIL) __Olden_panic("No more return messages\n");

  memcpy(&(__retmsgs[retmsgnum].retmsg),retMsg,size);
  /* Send retMsg with size to nodeId */
  CMAML_request(nodeId,__recv_msg,&(__retmsgs[retmsgnum].retmsg),size,
                MIGRATE_ON_RET+retmsgnum,source);
}

/* ReturnMsgRecv called from MsgDispatch and retaddr */
void __ReturnMsgRecv(sp,message,flag)
     word_t sp; /* placeholder-- called by NORETURN */
     retmsg_t *message;
     int flag; /* did we come from ourselves ? */
{
  /* copy the return structure to the correct place --
     check hidden argument in frame (get pointer from message) */
  /*chatting("return msg recv\n");*/
  if (message->retsize)
    memcpy(*((word_t *) (message->fp+64)), message->retstruct,
           message->retsize);

  /* Have received a return, fewer outstanding messages */
#ifdef VISUALIZE
  visProcMigRet();
#endif
  ThreadCount--;
  if (flag) /* we came from ourselves */
    {
      /*chatting("from ourselves\n");*/

      /* lazy scheme IV */
      /* here we actually haven't stored the modmask, but it isn't
         necessary as when we migrated here we cleared everything */
      
      memcpy(&(msg.ret),message,BASE_RETMSG_SZ);
      NORETURN(0,&(msg.ret),__MigrateReturn);
    }
  else 
    {
      /* lazy scheme IV */
      /* have sent across and back the modmask */
      __ModifiedMask[0] = message->modmask[0];
      __ModifiedMask[1] = message->modmask[1];
      __ClearCache2byMask(__ModifiedMask);
      
      NORETURN(0,message,__MigrateReturn);
    }
  /* Use placeholders for sp in NORETURN calls */
}  

