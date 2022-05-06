/* $Log: stack.asm,v $
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
 *
 */

#include "registers.h"
#include "constants.h"
#include "stack_frame.h"

.global ___setsp
.align 4
.proc 4

___setsp:
   jmpl %o7+8, %g0 /*  Note we sub 72 as frame ext size */
   sub %o0,72,%sp  /* delay slot */

.global ___getsp
.align 4
.proc 4

___getsp:
   jmpl %o7+8, %g0  /* Note we add 72 as this is size of frame extension */
   add %sp,72,%o0  /* delay slot */

.global ___getfp
.align 4
.proc 4

___getfp:
   jmpl %o7+8, %g0
   mov %fp, %o0  /* delay slot */

.global ___GoExpress
.align 4
.proc 4

___GoExpress:
   jmpl %o7+8, %g0
   or EXPRESSREG,EXPRESSMASK,EXPRESSREG      /* delay slot */

.global ___SlowLane
.align 4
.proc 4

___SlowLane:
   jmpl %o7+8, %g0
   andn EXPRESSREG,EXPRESSMASK,EXPRESSREG   /* delay slot */

.global ___InitRegs
.align 4
.proc 4

___InitRegs:
   mov %o0, IDREG            /* save processor name in IDREG */
   or EXPRESSREG,EXPRESSMASK,EXPRESSREG   /* set initially express */
   set ___PageTable,TBL_REG
   jmpl %o7+8, %g0
   mov %g0, FUTREG         /* delay slot - initialize FC ptr */

