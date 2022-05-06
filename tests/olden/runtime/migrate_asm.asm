/* $Log: migrate_asm.asm,v $
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
;; Revision 1.1  1995/09/13  18:59:52  mcc
;; Initial revision
;;
 *
 */

/* SPARC implementation of migration                    */
#include "constants.h"
#include "registers.h"
#include "stack_frame.h"
#include "call-stub.h"
#include "migrate.h"

.seg "text"
.global ___Migrate
.align 4
.proc 4

/* void ___Migrate(void *p, int frameSzB, int argsize, int retsize,
		   void *fnptr) */
/* CALLMSG offsets now off of the fp!! */
/* Here I store things in the frame b/c for reasons I don't understand the
   local regs occasionally lose their values */
___Migrate:
   add %sp, -168, %sp            /* alloc frame for migrate and extension for 
                                    call to CallMsgSend.  This
                                    consists of a 72 byte Migrate
                                    frame and 96 byte frame extension
                                    Migrate needs a frame as otherwise
                                    the message may be destroyed
                                    during the NORETURN in CallMsgSend */
   st %g0, [%sp+160]             /* set splink for migrate's frame */
   /* Place arguments in frame extension */
   st %o4, [%sp+92]              /* fnptr - 7th arg to CallMsgSend */

   sub %fp, %o1, %o4             /* message ptr - fp-frameSzB */

   /* set up migrate message */
   st %o7, [%fp+CALLMSG_PC]
   st %o0, [%fp+CALLMSG_P]
   st %o3, [%fp+CALLMSG_RETSIZE]
   st %o1, [%fp+CALLMSG_FRAMESIZE]
   st %o2, [%fp+CALLMSG_ARGSIZE]

   mov %o4, %o3              /* bp - 4th arg to CallMsgSend */
   set retaddr, %l0           
   ld [%fp-8], %l1
   xorcc %l0, %l1, %o4       /* is retaddr the return address of this fn */
   bne Migrate1; nop         /* flag - 5th arg to CallMsgSend */
   
   ld [%fp-4], %l0
   ld [%l0+CALLSTUB_NODEID], %l1
   ld [%l0+CALLSTUB_RETFP], %l2
   st %l1, [%fp+CALLMSG_NODEID]
   ba Migrate2
   st %l2, [%fp+CALLMSG_FP]   /* delay slot */


Migrate1:
   and IDREG,IDMASK,%o5
   st %o5, [%fp+CALLMSG_NODEID]
   st %fp, [%fp+CALLMSG_FP]

Migrate2:
   /* put callee-save registers into message */
   st %i0, [%fp+CSBUF_i0]
   st %i1, [%fp+CSBUF_i1]
   st %i2, [%fp+CSBUF_i2]
   st %i3, [%fp+CSBUF_i3]
   st %i4, [%fp+CSBUF_i4]
   st %i5, [%fp+CSBUF_i5]
                          /* We are going to put a zero in fc slot */
                          /* in the frame of the migrator */
   st %g0,[%fp+CSBUF_fc]  

   call ___CallMsgSend
   mov %fp, %o5              /* delay slot fp - 6th arg to CallMsgSend */

/* should never return here */


   
.global ___MigrateReturn
.align 4
.proc 4

/* void ___MigrateReturn (word_t sp, retmsg *msg) */

___MigrateReturn:
   mov %o0, %sp                   
   ld [%o1+RETMSG_FS], %l0
   ld [%o1+RETMSG_FP], %fp

   ld [%o1+RETMSG_i0], %i0
   ld [%o1+RETMSG_i1], %i1
   ld [%o1+RETMSG_i2], %i2
   ld [%o1+RETMSG_i3], %i3
   ld [%o1+RETMSG_i4], %i4
   ld [%o1+RETMSG_i5], %i5

   ld [%o1+RETMSG_o0], %o0
   ldd [%o1+RETMSG_FLRETURN], %f0

   ld [%fp-4], %l1                /* fp of caller (or fc ptr) */
   ld [%fp-8], %o7                /* return address */
   ld [%o1+RETMSG_ARGSIZE], %l3   
   sub %fp, %l0, %l2              /* sp of this frame  */

/* Fix future-cell stack if necessary */
/* If future stack empty, fill with value from frame */
   cmp %g0,FUTREG
   be,a ___MigrateRetFc; ld [%fp+CSBUF_fc],FUTREG
/* delay slot is taken only on branch -- i.e. stack empty */

___MigrateRetFc:
   cmp %l2, %sp  
   bne ___MigrateReturn.2
   add %l3, 64, %l4               /* delay slot */

   /* bottom frame */
   ld [%l4+%fp], %l2
   cmp %l2, %g0
   bne ___MigrateReturn.1
   add %fp, %l3, %sp              /* delay slot */

   jmpl %o7+12, %g0
   mov %l1, %fp

___MigrateReturn.1:
   ld [%sp+64], %l2
   cmp %l2, %g0
   bne,a ___MigrateReturn.1
   mov %l2, %sp                   /* delay slot */

   jmpl %o7+12, %g0
   mov %l1, %fp

___MigrateReturn.2:

   add %fp, %l3, %l3              /* fp+frameext */
   st %l3, [%l2+64]
   jmpl %o7+12, %g0
   mov %l1, %fp

.global ___SendRpc,___SendMultiRpc
.align 4
.proc 4

___SendMultiRpc:
   /* Only change is to add 2 to fcp -- by unaligning, we know that
      this is a multirpc, then fall through to ___SendRpc */
   add %l4,2,%l4
 
___SendRpc:
   /* At this point, we expect arguments to come in via normal method,
      but we require additional arguments (i.e. pn_to, rettype, 
      argcount,pc,fcp).  Use %l0-%l4 for these. */

   sll %l0,16,%l0 /* we are freeing %l0 */
   or %l1,%l0,%l1 /* l1 now [pn_to,rettype] */
   /* l0 now free -- compute size of stack extension */
   sll %l2,2,%l0  /* argcount * 4 */
   add %l0,75,%l0 
   srl %l0,3,%l0
   sll %l0,3,%l0   /* round to nearest 8, i.e. why we added +7 */
   add %l0,-120,%l0 /* f.e. size of 120 needed for 12 arguments */
   add %sp,%l0,%sp  /* allocate stack extension */
   sub %sp,%l0,%l0      /* save sp for future use */

   /* Relieve some register pressure */
   st %o5,[%sp+108] /* arg 6 */
   st %o4,[%sp+104] /* arg 5 */
   st %o3,[%sp+100] /* arg 4 */
   st %o2,[%sp+96]  /* arg 3 */
   st %o1,[%sp+92]  /* arg 2 */

   mov %o0,%o5 /* arg 1 */

   /* arg 7 */
   ld [%l0+92],%o2  
   st %o2,[%sp+112] 

   /* other arguments */
   srl %l1,16,%o0
   sll %l1,16,%o1
   srl %o1,16,%o1
   mov %l2,%o2
   mov %l3,%o3
   mov %l4,%o4
   
   ba ___RpcSend2; nop


