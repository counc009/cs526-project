/* $Log: cache_asm.asm,v $
;; Revision 1.2  1995/12/01  15:46:17  mcc
;; Added allocation on write.
;;
;; Revision 1.1  1995/09/22  17:34:36  mcc
;; Initial revision
;;
;; Revision 1.2  1995/09/19  15:08:46  mcc
;; Fixed bugs in abort/reverify
;;
;; Revision 1.1  1995/09/13  18:59:52  mcc
;; Initial revision
;;
 *
 */
.seg "text"

#include "registers.h"
#include "cache.h"
#include "constants.h"
#include "stack_frame.h"

/* AllocateOnWrite provides a spaghetti-stack interface for CWrite to
   Cache2 
 */

.global ___AllocateOnWrite
.align 4
.proc 4

___AllocateOnWrite:
   add %sp,-72,%sp
   st %o7,[%sp+68]
   call ___Cache2; st %g0,[%sp+64]
   add %sp,88,%sp /* we add 88 to compensate for -88 below */
   ld [%sp+68],%o7
   jmpl %o7+8,%g0; add %sp,72,%sp

/* Cache2 takes an argument which is an address.  First, it looks up
   this address in the page table (which is hashed).  We assume the
   address of this table is contained in a register (TBL_REG).  If
   the page is not present, we signal an allocation miss.  Otherwise,
   we look at the state information for the page to determine if the
   line is valid.  If so, we return its address, otherwise we have a
   cache line miss. */
/* addresses are of form:[14 bit page #|11 bit page offset|7 bit pn] */

.global ___Cache2
.align 4
.proc 4

___Cache2:
   and %o0,IDMASK,%l0 /* get proc id -- l0*/
   srl %o0,7,%l1 /* shift off pn */
   and %l1,(PAGE_SIZE-1),%l1 /* get page offset -- l1 */
   srl %o0,18,%l2 /* get page # */
   and %l2,(PG_TBL_SZ-1),%l3 /* get abbrev page # */
   sll %l0,3,%l4    /* l4 has proc id << 3 */
   xor %l3,%l4,%l3  /* get hash value */
#ifdef VISUALIZE
   mov %l3,%g2
#endif
   sll %l3,4,%l3 /* multiply by 16 --l3 has hash*16 */
   sll %l2,7,%l2 /* l2 had page # now has page # << 7 */
   or %l2,%l0,%l2  /* compute tag [14 bit page # | 7 bit pn] */
   ld [TBL_REG+%l3],%l4 /* get tag -- l4 */
   cmp %l4,%l2          /* check tag */
   be ___PageHit; add TBL_REG,%l3,%l3 /* place entry pointer in l3 */

   /* At this point, we want to walk down linked list */ 
   /* tag is in l2, page offset is in l1, entry pointer in l3 */
___PageLoop:
   ld [%l3+NEXT_OFFSET],%l3 /* get entry pointer */
   cmp %l3,%g0
   be,a ___PageMiss; add %sp,-88,%sp
   /* use max stack offset for all-- coordinate with pseudos.c */
   /* If null pointer, goto page miss-- allocate f.e. */
   ld [%l3+TAG_OFFSET],%l4     /* get tag -- l4 */
   cmp %l4,%l2      /* check tag */
   bne ___PageLoop; nop
   /* must be followed by ___PageHit */

___PageHit:
   /* require page offset to be in %l1, entry pointer in l3 */
   ld [%l3+PAGE_OFFSET],%l0  /* get location of page from entry -- l0*/
   ld [%l3+VALID_OFFSET],%l2  /* load state information for page */
   srl %l1,6,%l4     /* get cache line offset -- 64 byte line */
   srl %l2,%l4,%l2   /* place bit in LSB */
   andcc %l2,1,%g0   /* Check valid bit for line */
   be ___SignalLineMiss; nop /* zero if not valid */
#ifdef VISUALIZE
   mov %o7,%l2
   mov %g2,%o0
   call _visCacheHit; add %sp,-72,%sp
   add %sp,72,%sp
   mov %l2,%o7
#endif
   jmpl %o7+12,%g0; add %l1,%l0,%o0 /* add page offset to page addr*/

___SignalLineMiss:
   /* At this point, we wish to signal cache line miss */
   /* require page to be in l0, page offset in l1, entry in l3,
      cache line offset in l4 */
   add %l3,VALID_OFFSET,%o1  /* second arg -- page valid bits location */
   mov %l4,%o2  /* third arg -- cache line offset */
   /* Warning!  pages are not page aligned -- align before adding */
   andn %l1,63,%o3 /* eliminate offset for lineaddr */
   add %o3,%l0,%o3 /* location of page + lineaddr offset (l1) */
   ba ___LineMiss; add %sp,-88,%sp

/* Cache2Lookup takes an argument which is an address.  First, it looks up
   this address in the page table (which is hashed).  We must first
   load address of cache table into a register (TBL_REG).  If
   the page is not present, we return 0.  Otherwise,
   we look at the state information for the page to determine if the
   line is valid.  If so, we return its address, otherwise we have a
   cache line miss and return 0. */
/* addresses are of form:[14 bit page #|11 bit page offset|7 bit pn] */

.global ___Cache2Lookup
.align 4
.proc 4

___Cache2Lookup:
   set ___PageTable,TBL_REG
   and %o0,IDMASK,%l0 /* get proc id -- l0*/
   srl %o0,7,%l1 /* shift off pn */
   and %l1,(PAGE_SIZE-1),%l1 /* get page offset -- l1 */
   srl %o0,18,%l2 /* get page # */
   and %l2,(PG_TBL_SZ-1),%l3 /* get abbrev page # */
   sll %l0,3,%l4    /* l4 has proc id << 3 */
   xor %l3,%l4,%l3  /* get hash value */
   sll %l3,4,%l3 /* multiply by 16 --l3 has hash*16 */
   sll %l2,7,%l2 /* l2 had page # now has page # << 7 */
   or %l2,%l0,%l2  /* compute tag [14 bit page # | 7 bit pn] */
   ld [TBL_REG+%l3],%l4 /* get tag -- l4 */
   cmp %l4,%l2          /* check tag */
   be ___LPageHit; add TBL_REG,%l3,%l3 /* place entry pointer in l3 */

   /* At this point, we want to walk down linked list */ 
   /* tag is in l2, page offset is in l1, entry pointer in l3 */
___LPageLoop:
   ld [%l3+NEXT_OFFSET],%l3 /* get entry pointer */
   cmp %l3,%g0
   be,a ___LSignalMiss; nop
   /* use max stack offset for all-- coordinate with pseudos.c */
   /* If null pointer, goto page miss-- allocate f.e. */
   ld [%l3+TAG_OFFSET],%l4     /* get tag -- l4 */
   cmp %l4,%l2      /* check tag */
   bne ___LPageLoop; nop
   /* must be followed by ___PageHit */

___LPageHit:
   /* require page offset to be in %l1, entry pointer in l3 */
   ld [%l3+PAGE_OFFSET],%l0  /* get location of page from entry -- l0*/
   ld [%l3+VALID_OFFSET],%l2  /* load state information for page */
   srl %l1,6,%l4     /* get cache line offset -- 64 byte line */
   srl %l2,%l4,%l2   /* place bit in LSB */
   andcc %l2,1,%g0   /* Check valid bit for line */
   be ___LSignalMiss; nop /* zero if not valid */
   jmpl %o7+8,%g0; add %l1,%l0,%o0 /* add page offset to page addr*/

___LSignalMiss:
   jmpl %o7+8,%g0; mov %g0,%o0 /* return 0 */

/* Invalid2 takes an argument which is an address.  First, it looks up
   this address in the page table (which is hashed).  We assume the
   address of this table is contained in a register (TBL_REG).  If
   the page is not present, we do nothing.  Otherwise,
   we look at the state information for the page to set the 
   line invalid.  */
/* addresses are of form:[14 bit page #|11 bit page offset|7 bit pn] */
.global ___Invalid2
.align 4
.proc 4

___Invalid2:
   and %o0,IDMASK,%l0 /* get proc id -- l0*/
   srl %o0,7,%l1 /* shift off pn */
   and %l1,(PAGE_SIZE-1),%l1 /* get page offset -- l1 */
   srl %o0,18,%l2 /* get page # */
   and %l2,(PG_TBL_SZ-1),%l3 /* get abbrev page # */
   sll %l0,3,%l4    /* l4 has proc id << 3 */
   xor %l3,%l4,%l3  /* get hash value */
   sll %l3,4,%l3 /* multiply by 16 --l3 has hash*16 */
   sll %l2,7,%l2 /* l2 had page # now has page # << 7 */
   or %l2,%l0,%l2  /* compute tag [14 bit page # | 7 bit pn] */
   ld [TBL_REG+%l3],%l4 /* get tag -- l4 */
   cmp %l4,%l2          /* check tag */
   be ___Invalid2PageHit; add TBL_REG,%l3,%l3 /* place entry pointer in l3 */

   /* At this point, we want to walk down linked list */ 
   /* tag is in l2, page offset is in l1, entry pointer in l3 */
___Invalid2PageLoop:
   ld [%l3+NEXT_OFFSET],%l3 /* get entry pointer */
   cmp %l3,%g0
   be,a ___GoBack; nop
   /* use max stack offset for all-- coordinate with pseudos.c */
   /* If null pointer, goto page miss-- allocate f.e. */
   ld [%l3+TAG_OFFSET],%l4     /* get tag -- l4 */
   cmp %l4,%l2      /* check tag */
   bne ___Invalid2PageLoop; nop
   /* must be followed by ___Invalid2PageHit */

___Invalid2PageHit:
   /* page offset to be in %l1, entry pointer in l3 */
   ld [%l3+VALID_OFFSET],%l2  /* load state information for page */
   srl %l1,6,%l4     /* get cache line offset -- 64 byte line */
   set 1,%l1
   sll %l1,%l4,%l1   /* have one in correct place */
   andn %l2,%l1,%l2
   st %l2,[%l3+VALID_OFFSET]
___GoBack:
   jmpl %o7+12,%g0; nop

