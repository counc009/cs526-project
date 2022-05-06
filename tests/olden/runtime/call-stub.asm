/* SPARC implementation of CallStub code */
/* $Log: call-stub.asm,v $
;; Revision 1.2  1995/12/18  21:24:32  mcc
;; Fixed bug-- RpcRetAddr on MakeCoherent, losing value of %o0
;;
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
;; Revision 1.1  1995/09/13  18:59:52  mcc
;; Initial revision
;;
 *
 */

#include "constants.h"
#include "registers.h"
#include "call-stub.h"
#include "migrate.h"
#include "stack_frame.h"
#include "rpc.h"

/* This code assumes that the necessary stack space has already been 
   allocated.   */

/* size 2 = frameSzB+EXTSZ+64 */

.seg "text"
.global ___CallStub,retaddr
.align 4
.proc 4

/* Moved CallMsg info to above fp */
/* void ___CallStub(word_t *CallMsg, word_t *sp, int framesize, int size2) */

___CallStub:
   /* Following mov removes frame extension allocated by msg.c */
   /* l0 -- stubsp
      l1 -- retmsg ptr
      l2 -- stubfp
      l3 -- message info */
   mov %o1, %sp
   add %o3, %sp, %l0           /* stubsp = sp+frameSzB+EXTSZ+64 */
   add %o2, %sp, %fp           /* fp = sp+frameSzB+64           */
   add %fp, 64, %fp

   add %l0, CALLSTUB_SZ,  %l1           /* retmsg ptr= stubsp+72     */
   add %l0, CALLSTUB_SZ,  %l2           /* stubfp = stubsp+72        */
   
   add %o0,%o2,%l3                     /* find actual message info */
   ld [%l3+CALLMSG_RETSIZE], %i0
   ld [%l3+CALLMSG_FP], %l4
   ld [%l3+CALLMSG_FRAMESIZE], %i1
   ld [%l3+CALLMSG_NODEID], %i2
   ld [%l3+CALLMSG_ARGSIZE], %i3


   st %i0, [%l1+RETMSG_RETSIZE]
   st %l4, [%l1+RETMSG_FP]
   st %i1, [%l1+RETMSG_FS]
   st %i3, [%l1+RETMSG_ARGSIZE]

   st %i2, [%l2+CALLSTUB_NODEID]
   
   st %g0, [%l0+64]               /*  mark stubframe as live       */
   st %g0, [%sp+128]              /*  mark migrated frame as live  */

   cmp %g0, %i0
   be noretstruct
   add %g0, %g0, %g0             /* delay slot - nop              */

   add RETMSG_RETSTRUCT, %l1, %l4
   st %l4, [%fp+64]            /* reset hidden parameter to point into
                                  retmsg space */

noretstruct:
   ld [%fp+CSBUF_i0], %i0
   ld [%fp+CSBUF_i1], %i1
   ld [%fp+CSBUF_i2], %i2
   ld [%fp+CSBUF_i3], %i3
   ld [%fp+CSBUF_i4], %i4
   ld [%fp+CSBUF_i5], %i5


   set retaddr, %o7            /* set new return address for restart */
   st %o7, [%fp-8]             /* of the migrated function           */
   st %l2, [%fp-4]             /* fp link                            */
   ld [%l3+CALLMSG_PC], %l4
   ld [%l3+CALLMSG_P], %o0
   srl %o0, PN_BITS, %o0         /* blast away node id from ptr        */
   jmpl %l4+8, %g0             /* branch to migrated function        */
   add %sp, 64, %sp            /* deallocate extra rwo space         */

retaddr:
   add %g0, %g0, %g0           /* nop */
   add %g0, %g0, %g0           /* nop */
   add %g0, %g0, %g0           /* nop */
   mov %fp, %l0                /* retmsgptr                          */

   st %i0, [%l0+RETMSG_i0]
   st %i1, [%l0+RETMSG_i1]
   st %i2, [%l0+RETMSG_i2]
   st %i3, [%l0+RETMSG_i3]
   st %i4, [%l0+RETMSG_i4]
   st %i5, [%l0+RETMSG_i5]

   st %o0, [%l0+RETMSG_o0]     /* store return value into frame      */
   std %f0, [%l0+RETMSG_FLRETURN]
   ld [%fp+CALLSTUB_NODEID], %o0

   /* set splink to deactivate the migrated frame, callstub frame, and
      return message.  splink points to the bottom of the frame
      above not to the splink of the frame above */
   add RETMSG_SIZE, %l0, %l1  
   sub %fp, 8, %l2            /* stubsp link field */
   st %l1, [%l2]

   and IDREG,IDMASK,%o1       /* strip off express checkout bit */
   cmp %o0, %o1
   bne retdiff
   add %g0, %g0, %g0          /* delay slot - nop */


   mov %sp, %o0                 /* sp - 1st arg Return Message Receive */
   mov %l0, %o1                 /* msg ptr - 2nd arg Return Message Receive */
   add %g0,1,%o2                /* flag - 3rd arg RMR */
   call ___ReturnMsgRecv
   sub %sp, 80, %sp             /* delay slot - allocate frame ext */
   /* will never return here */

retdiff:
   /* o0 already contains processor name */
   sub %sp, 80, %sp             /* allocate frame ext */
   call ___CallMsgReply
   mov %l0, %o1                 /* delay slot - msgptr - 2nd arg */
   nop                          /* +12 hack */

   /* does the deallocation code, deallocate the frame ext? */
   mov %g0,%o0 /* We want to assert that future stack is empty */
   call ___MsgDispatch
   sub %sp, 80, %sp             /* delay slot - allocate frame ext */
   /* will never return here */

.global ___RpcStub
.align 4
.proc 4

/* void ___RpcStub(sp,RpcMsg,mcb) */

/* Stub frame has id, from */
___RpcStub:
  /* i0 = ptr
     i1 = index
     i2 = argc
     i3 = PC
     i4 = frame ext sz
   */
  ldub [%o1],%l0 /* load rettype */
  ldub [%o1+1],%l1 /* load argc */
  lduh [%o1+2],%l2 /* load from */
  ld [%o1+4],%l3 /* load PC */
  ld [%o1+8],%i0 /* load fcp */
  add %g0,75,%i4 /* 68 + 7 for rounding */
  sll %l1,2,%l1
  add %i4,%l1,%i4
  srl %i4,3,%i4
  sll %i4,3,%i4  /* size of stack frame extension */
  add %o0,-RPCSTUBSZ,%sp /* o0 has NORETURN sp -- used to dealloc f.e. */
  cmp %l0,4
  bl ___RpcStub2; mov %g0,%l4 /* no return size */
  mov %l0,%l4
  sub %sp,%l4,%sp /* allocate space for structure return */
___RpcStub2: /* now have size of return value in g4 */
  add %sp,RPCSTUBSZ,%fp       /* set frame pointer */
  st %l4, [%fp-STRSZ]     /* store structure size */
  st %l0, [%fp-RETTYPELOC]        /* store rettype */
  st %i0, [%fp-FCPLOC]   /* store fcp */
  st %l2, [%fp-FROMLOC]   /* store from */
  st %g0, [%sp+64]       /* spaghetti stack */
  mov %o1,%i0 /* Save pointer */
  mov %o2,%i1 /* save integer index */
  mov %l1,%i2 /* save argument count */
  mov %l3,%i3 /* save PC */
  /* All l regs free at this point */
  /* load arguments */
  ld [%i0+12],%o0
  ld [%i0+16],%o1
  ld [%i0+20],%o2
  ld [%i0+24],%o3
  ld [%i0+28],%o4
  ld [%i0+32],%o5
  ld [%i0+36],%l0
  /* save arguments */
  st %o0,[%fp-ARGV]
  st %o1,[%fp-ARGV+4]
  st %o2,[%fp-ARGV+8]
  st %o3,[%fp-ARGV+12]
  st %o4,[%fp-ARGV+16]
  st %o5,[%fp-ARGV+20]
  st %l0,[%fp-ARGV+24]

  /* Free the Rpc for later use */
  add %sp,-72,%sp
  call ___RpcFree; mov %i1,%o0
  add %sp,72,%sp

  sub %sp,%i4,%sp        /* allocate frame extension */
  st %fp, [%sp+64]       /* set hidden argument */

  /* load arguments */
  cmp %i2,6  /* check arg count */
  ble ___RpcStub3; ld [%fp-ARGV],%o0
  ld [%fp-ARGV+24],%l0
  st %l0,[%sp+92]
___RpcStub3:
  ld [%fp-ARGV+4],%o1
  ld [%fp-ARGV+8],%o2
  ld [%fp-ARGV+12],%o3
  ld [%fp-ARGV+16],%o4
  ld [%fp-ARGV+20],%o5

  jmpl %i3,%o7; nop
  nop /* must have this compiled with -m or bad things happen here */
/* now we return from the Rpc */
/* All registers free here */
___RpcRetAddr:
  std %f0,[%fp-DBLRETLOC]
  mov %o0,%i0 /* save in CSR */
  call ___MakeCoherent; add %sp,-80,%sp
  add %sp,80,%sp
  ld [%fp-RETTYPELOC],%o2
  cmp %o2,1
  be ___RetValComplete; mov %i0,%o4
  cmp %o2,2
  be ___RetValComplete; ld [%fp-DBLRETLOC],%o4
  cmp %o2,3
  be ___RetValComplete; ld [%fp-DBLRETLOC+4],%o5
  ld [%fp],%o4
  ld [%fp+4],%o5
___RetValComplete:
  ld [%fp-FROMLOC],%o0
  set ___RpcReplyHandler, %o1
  ld [%fp-FCPLOC],%o3
  call _CMAML_request; add %sp,-96,%sp
  add %sp,96,%sp

  ld [%fp-RETTYPELOC],%o2
  cmp %o2,8
  ble ___CleanupRpc; nop

/* Currently can only handle up to 8 byte return values */
/* Insert code here for larger return values */
  set Lamr1,%o0
  call ___Olden_panic; add %sp,-72,%sp

___CleanupRpc:
/* clean up the stub frame here */
  ld [%fp-STRSZ],%o1
  sub %fp,RPCSTUBSZ,%o0
  cmp %o0,%sp
  add %o1,RPCSTUBSZ,%o2
  be,a ENDreply; add %sp,%o2,%sp
  add %fp,%o1,%o1
  st %o1,[%fp-SPLINKLOC]
ENDreply:
  add %sp,-80,%sp
  call ___MsgDispatch; /* delay slot below */
           mov %g0,%o0 /* we want to assert future stack empty */


.seg "text"
Lamr1:.ascii "Got to here\012\000"
Lamr3:.ascii "p = %x\012\000"


   


