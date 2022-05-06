/* $Log: stack_frame.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 * Here is where stack frame locations are reserved
 * also stuff dealing with spaghetti stack, and initializing regs.
 */

/* Note -4 has fp, -8 has return address */
#define CSBUF_fc -12 
#define CSBUF_BT -16
#define CSBUF_BASE        -20
#define CSBUF_i0  (CSBUF_BASE-0)
#define CSBUF_i1  (CSBUF_BASE-4)
#define CSBUF_i2  (CSBUF_BASE-8)
#define CSBUF_i3  (CSBUF_BASE-12)
#define CSBUF_i4  (CSBUF_BASE-16)
#define CSBUF_i5  (CSBUF_BASE-20)

#define EXPRESSMASK 128 /* leave 7 bits for proc id */
   
