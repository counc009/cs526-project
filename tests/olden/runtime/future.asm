/* $Log: future.asm,v $
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
;; Revision 1.2  1995/09/19  15:09:31  mcc
;; Fixed bug in abort routine.
;;
;; Revision 1.1  1995/09/13  18:59:52  mcc
;; Initial revision
;;
 *
 */

#include "constants.h"
#include "registers.h"
#include "future-cell.h"
#include "stack_frame.h"
#include "threads.h"
#include "runtime.h"

.seg "text"
.global ___TopFutureStk
.align 4
.proc 4

___TopFutureStk:
   jmpl %o7+8, %g0
   mov FUTREG, %o0         /* delay slot */

.global ___PopFutureStk
.align 4
.proc 4

___PopFutureStk:
   mov FUTREG, %o0
   cmp FUTREG, %g0
   be nopop
   add %g0, %g0, %g0             /* delay slot - nop */

   jmpl %o7+8, %g0
   ld [FUTREG+FCELL_NEXT], FUTREG      /* delay slot - pop! */

nopop:
   jmpl %o7+8, %g0
   add %g0, %g0, %g0             /* delay slot - nop */

.global ___RpcStolen,___stolen
.align 4
.proc 4

/* first argument is sp, second is future cell ptr */
___RpcStolen:
   mov %o1,%fp
   mov %o0,%sp

___stolen:
/* assume that the future cell ptr comes in %fp */

   ld [FCELL_NEXT+%fp], %l0

   cmp %l0,FC_WAITING /* could be either stolen or waiting */
   be ___resume1
   add %g0, %g0, %g0          /* delay slot - nop */

   /* Don't need to mark full, unless it is not yet waiting */
   mov FC_FULLRET, %l1
   st %l1, [FCELL_NEXT+%fp]   /* state encoded in next field, fc opt */

   /* lazy scheme IV -- store mask in FC */
   set ___ModifiedMask,%l1
   ld [%l1],%l2
   ld [%l1+4],%l3
   st %l2,[%fp+FCELL_MODMASK0]
   st %l3,[%fp+FCELL_MODMASK1]

   mov %g0, %o0               /* assert future stack empty */
   call ___MsgDispatch        /* ___MsgDispatch(sp, fp, fnptr) */
   add %sp, -80, %sp          /* delay slot - frame ext for MsgDispatch */
                              /* MsgDispatch will not use either
                                 fp or fnptr in this case      */
___resume1:

   /* lazy scheme IV -- do before reloading fp */
   /* OR together two masks, one from ret msg, one stored at suspend */
   set ___ModifiedMask, %i0
   ld [%i0],%l0
   ld [%i0+4],%l1
   ld [%fp+FCELL_MODMASK0],%i1
   ld [%fp+FCELL_MODMASK1],%i2
   or %l0,%i1,%i1
   or %l1,%i2,%i2
   st %i1,[%i0]
   st %i2,[%i0+4]

   ld [%fp+FBUF_PC], %o7
   ld [%fp+FBUF_FP], %fp

   cmp %g0,FUTREG
   be,a ___resume3; ld [%fp+CSBUF_fc],FUTREG
___resume3:
   ld [%fp+CSBUF_i0], %i0
   ld [%fp+CSBUF_i1], %i1
   ld [%fp+CSBUF_i2], %i2
   ld [%fp+CSBUF_i3], %i3
   ld [%fp+CSBUF_i4], %i4
   jmpl %o7+8, %g0         
   ld [%fp+CSBUF_i5], %i5    /* delay slot */


.global ___MultiSuspend
.align 4
.proc 4

___MultiSuspend:
   st %o7, [%o0+MCELL_PC] /* state as well as pc */

   st %g0, [%fp+CSBUF_fc]
   st %i0, [%fp+CSBUF_i0]
   st %i1, [%fp+CSBUF_i1]
   st %i2, [%fp+CSBUF_i2]
   st %i3, [%fp+CSBUF_i3]
   st %i4, [%fp+CSBUF_i4]
   st %i5, [%fp+CSBUF_i5]

   mov %fp, %o1
   mov %sp, %o0                /* Don't assert fut stk empty */
   call ___MsgDispatch         /* ___MsgDispatch(sp, fp, fnptr) */
   add %sp, -80, %sp           /* frame extension for MsgDispatch */

.global ___Suspend
.align 4
.proc 4

/* ___Suspend(future_cell_t *cell, void, void *fnptr) */
/* Be careful not to use o1,o2 as they are passed to MsgD */

___Suspend:
   sethi %hi(___ModifiedMask),%o3
   ld [%o0+FCELL_NEXT],%o4
   cmp %o4,FC_FULLRET
   bne ___Suspend2; or %o3,%lo(___ModifiedMask),%o3
   
   /* lazy scheme IV -- or FC mask with ModifiedMask */
   ld [%o3],%o4
   ld [%o3+4],%o5
   ld [%o0+FCELL_MODMASK0],%l0
   ld [%o0+FCELL_MODMASK1],%l1
   or %o4,%l0,%l0
   st %l0,[%o3]
   or %o5,%l1,%l1
   st %l1,[%o3+4]

   jmpl %o7+8,%g0; nop

___Suspend2:
   st %o7, [%o0+FBUF_PC]
   st %fp, [%o0+FBUF_FP]

   st %g0, [%fp+CSBUF_fc]
   st %i0, [%fp+CSBUF_i0]
   st %i1, [%fp+CSBUF_i1]
   st %i2, [%fp+CSBUF_i2]
   st %i3, [%fp+CSBUF_i3]
   st %i4, [%fp+CSBUF_i4]
   st %i5, [%fp+CSBUF_i5]

   /* lazy scheme IV */
   /*set ___ModifiedMask, %o3*/ /* see above */
   ld [%o3],%o4
   ld [%o3+4],%o5
   st %o4,[%o0+FCELL_MODMASK0]
   st %o5,[%o0+FCELL_MODMASK1]

   mov FC_WAITING, %o1
   st %o1, [%o0+FCELL_NEXT]    /* Store WAITING in NEXT field, fc opt */
   mov %fp, %o1
   mov %sp, %o0                /* Don't assert fut stk empty */
   call ___MsgDispatch         /* ___MsgDispatch(sp, fp, fnptr) */
   add %sp, -80, %sp           /* frame extension for MsgDispatch */

/* __MultiSteal(char *sp, mcell *mc) */

.global ___MultiSteal
.align 4
.proc 4

___MultiSteal:
  ld [%o1+MCELL_FP],%fp
  ld [%o1+MCELL_STEALADDR],%o7

  /* reload callee-saves */
  ld [%fp+CSBUF_i0],%i0
  ld [%fp+CSBUF_i1],%i1
  ld [%fp+CSBUF_i2],%i2
  ld [%fp+CSBUF_i3],%i3
  ld [%fp+CSBUF_i4],%i4
  ld [%fp+CSBUF_i5],%i5
  
  /* storing PC directly before call */
  jmpl %o7+0,%g0; mov %o0,%sp

.global ___Steal
.align 4
.proc 4

/* ___Steal(void *sp, future_cell *fc, void *fp, void *fnptr) */
/* this code assumes that the callee save unrolling code will
   not use registers l0, o4, and o5.                          */

/* o4 - fp  SET FPREG to this in runtime.h
   o5 - fp of caller 
   l0 - ret addr 
   l4 - fcp
 */
___Steal:
   mov %o1, %l4                     /* save future cell ptr */
   mov %o2, %o4                     /* this fn's fp */
   mov FC_STOLEN, %l0
   st %l0,[%l4+FCELL_NEXT]         /* Mark as stolen in NEXT field, fc opt */

___Steal2:
   ld [%o4-8], %l0                  /* this fn's ra */
   jmpl %o3, %o7                    /* do this fn's callee-save unrolling */
   ld [%o4-4], %o5                  /* fp of this fn's caller */
   cmp %o5, %l4                     /* are we at the frame of the future? */
   bgu ___Steal3                   /* we are if the fp is greater than the 
                                       fcptr */
   add %g0, %g0, %g0              /* delay slot - nop */

   ld [%l0+4], %o3              /* callee-save code ptr for this fn's caller */
   ba ___Steal2
   mov %o5, %o4                     /* delay slot - follow the fp link */

___Steal3:
   mov %o5, %fp                     
   add OFFSET2, %l0, %l3            /* future starting code */
   add -OFFSET, %l0, %l0            /* new return address   */
   st %l0, [%o4-8]                  /* store new return address in the frame of
                                       the future call's callee */
   st %o1, [%o4-4]                  /* patch the frame pointer with a pointer to
                                       the future cell */
   jmpl %l3+12, %g0
   mov %o0, %sp                      /* delay slot  */
                                       
.global ___Overflow
.align 4
.proc 4

___Overflow:
/* void __Overflow(sp,fc,fp) 
    *(fp+CSBUF_fc)=fc;
   FCPTR=0;
   __MsgDispatch(sp,0,0);  ! fc and fnptr unneeded 
   } */
  mov %o0,%sp /* NORETURN called us */
  st %o1,[%o2+CSBUF_fc]
  mov %g0,FUTREG
  /* Don't need to assert on fut stk as we just set it (arg %o0) */
  call ___MsgDispatch; add %sp,-80,%sp
   
.global ___GetRetAddr
.align 4
.proc 4

___GetRetAddr:
  jmpl %o7+8,%g0; ld [%fp-8],%o0

.global ___OverflowRPC
.align 4
.proc 4

/* void __OverflowRPC(sp,fcp) */
___OverflowRPC:
  /* frame pointer restored by NORETURN */
  mov %o0,%sp
  st FUTREG,[%o1+FCELL_NEXT]
  st %g0,[%fp+CSBUF_fc]
  st %fp,[%o1+FBUF_FP]
  st %i0,[%fp+CSBUF_i0]
  st %i1,[%fp+CSBUF_i1]
  st %i2,[%fp+CSBUF_i2]
  st %i3,[%fp+CSBUF_i3]
  st %i4,[%fp+CSBUF_i4]
  st %i5,[%fp+CSBUF_i5]
  mov %g0,FUTREG
  /* Don't need to assert on fut stk as we just set it (arg %o0) */
  call ___MsgDispatch; add %sp,-80,%sp

.global ___RpcContinue
.align 4
.proc 4

/* void __RpcContinue(sp,fcp) */
___RpcContinue:
  ld [%o1+FBUF_FP],%fp
  ld [%o1+FBUF_PC],%o7
  ld [%o1+FCELL_NEXT],FUTREG
  ld [%fp+CSBUF_i0],%i0
  ld [%fp+CSBUF_i1],%i1
  ld [%fp+CSBUF_i2],%i2
  ld [%fp+CSBUF_i3],%i3
  ld [%fp+CSBUF_i4],%i4
  ld [%fp+CSBUF_i5],%i5
  jmpl %o7+RPCOFFSET,%g0; mov %o0,%sp
  
/* void __MultiStolen(mc) */
.global ___MultiStolen
.align 4
.proc 4

___MultiStolen:
  ld [%o0+MCELL_COUNT],%l0
  subcc %l0,1,%l0
  bne ___MS2; mov %o0,%o1 /* if count not zero-- dispatch */
  ld [%o1+MCELL_PC],%l1
  cmp %l1,%g0
  be ___MS2; nop /* if zero, haven't touched yet */
  ba ___MultiRpcResume; mov %sp,%o0 /* o/w resume */
___MS2: /* don't resume */
  /* store decremented count */
  st %l0,[%o1+MCELL_COUNT]
  mov %fp,%o1
  ld [%o7+8],%o2
  add %sp,-80,%sp
  call ___MsgDispatch; mov %sp,%o0
  /* MsgD(sp,fp,fnptr) */
   
.global ___MultiRpcResume
.align 4
.proc 4

/* void __MultiRpcResume(sp,mc) */
___MultiRpcResume:
  ld [%o1+MCELL_FP],%fp
  ld [%o1+MCELL_PC],%o7
  ld [%fp+CSBUF_i0],%i0
  ld [%fp+CSBUF_i1],%i1
  ld [%fp+CSBUF_i2],%i2
  ld [%fp+CSBUF_i3],%i3
  ld [%fp+CSBUF_i4],%i4
  ld [%fp+CSBUF_i5],%i5
  jmpl %o7+8,%g0; mov %o0,%sp

.global ___ThreadSuspend
.align 4
.proc 4

/* __ThreadSuspend(&bt) */
___ThreadSuspend:

   st %o7, [%o0+BARRIER_PC] /* state as well as pc */
   st %fp, [%o0+BARRIER_FP]

   ld [%o7+4],%o1 /* fnptr -- arg 2 */
   mov %fp,%o2 /* fp -- arg 3 */
   st %g0, [%fp+CSBUF_fc]
   st %i0, [%fp+CSBUF_i0]
   st %i1, [%fp+CSBUF_i1]
   st %i2, [%fp+CSBUF_i2]
   st %i3, [%fp+CSBUF_i3]
   st %i4, [%fp+CSBUF_i4]
   st %i5, [%fp+CSBUF_i5]

   /* for 3 arguments (rather than 1), need 8 more in f.e. */
   ba ___ThreadSuspend2; add %sp,-8,%sp

.global ___MultiBarrier,___BarrierResume
.align 4
.proc 4

/* __MultiBarrier(&bc, &mc, f, arg1, arg2) */
___MultiBarrier:
   ld [%o7+4],%o5 /* fnptr -- arg 6 */
   st %o7, [%o0+BARRIER_PC] /* state as well as pc */
   st %fp, [%o0+BARRIER_FP]

   st %g0, [%fp+CSBUF_fc]
   st %i0, [%fp+CSBUF_i0]
   st %i1, [%fp+CSBUF_i1]
   st %i2, [%fp+CSBUF_i2]
   st %i3, [%fp+CSBUF_i3]
   st %i4, [%fp+CSBUF_i4]
   st %i5, [%fp+CSBUF_i5]

   /* for 7 arguments (rather than 5), need 8 more in f.e. */
   add %sp,-8,%sp

   ba ___Barrier2; st %fp, [%sp+92] /* arg 7 */


/* __BarrierResume(sp,&bc) */
___BarrierResume:
   ld [%o1+BARRIER_FP],%fp
   ld [%o1+BARRIER_PC],%o7
   st %o0,[%o1+BARRIER_SP]

   ld [%fp+CSBUF_i0],%i0  
   ld [%fp+CSBUF_i1],%i1  
   ld [%fp+CSBUF_i2],%i2  
   ld [%fp+CSBUF_i3],%i3  
   ld [%fp+CSBUF_i4],%i4  
   ld [%fp+CSBUF_i5],%i5  
   jmpl %o7+12,%g0; mov %o0,%sp

