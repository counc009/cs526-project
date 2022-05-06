/* $Log: plain2.asm,v $
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
;; Revision 1.1  1995/09/13  18:59:52  mcc
;; Initial revision
;;
 *
 */

#include "registers.h"
#include "future-cell.h"
#include "runtime.h"

.seg "text"
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

.global ___InitRegs
.align 4
.proc 4

___InitRegs:
   mov %o0, IDREG            /* save processor name in IDREG */
                           /* will need to be fixed once we
                              know what SPARC heap addresses
                              look like */
   or EXPRESSREG,EXPRESSMASK,EXPRESSREG     /* set initially happy */
   jmpl %o7+8, %g0
   mov %g0, FUTREG         /* delay slot - initialize FC ptr */


.global ___stolen
.align 4
.proc 4

___stolen:
/* assume that the future cell ptr comes in %fp */

   jmpl %o7+8, %g0         
   ld [%fp+CSBUF_i5], %i5    /* delay slot */

.global ___Suspend,___CWrite2D,___CWrite2,___Cache2,___Migrate
.global ___SendRpc,___ClearCache2,___Invalid2
.align 4
.proc 4

/* ___Suspend(future_cell_t *cell, void, void *fnptr) */

___SendRpc:
___ClearCache2:
___Invalid2:
___CWrite2D:
___CWrite2:
___Cache2:
___Migrate:
___Suspend:

   jmpl %o7+8, %g0; nop
