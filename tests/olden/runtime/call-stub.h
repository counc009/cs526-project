
/* $Log: call-stub.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

/* sparc implementation of call-stub.h  */


#ifndef _CALL_STUB_
#define _CALL_STUB_

/* The stack layout of the call-stub's frame is:
 *
 *              | retmsg     | <-- stub FP    <-- retmsg ptr
 *              +============+ 
 *     -4:      |  nodeId    |               
 *              +------------+               
 *     -8:      |  splink    |  
 *              +------------|
 *              |  64 bytes  |
 *              |  for rwo   | <-- stub SP
 *              +============+
 *              |  frame ext |
 *              |  for       |
 *              |  migrated  |
 *              |  frame     | <-- FP
 *              +============+
 *     -4:      |  stub FP   |                ^
 *              +------------+                |
 *     -8:      |  stub PC   |                |
 *              +------------+             migrated frame
 *    -12       |            |                |
 *              |  cs buf    |                |
 *              |            |                |
 *              +------------+                |
 *    -36:      |            |                |
 *              |  locals    |                |
 *              +------------+                |
 *              |  splink    |                |
 *              +------------+                |
 *              |  message   |                |
 *              |  buffer +  |                |
 *              |  36 bytes  |                |
 *              |  of dead   |                |
 *              |  space     | <- SP (real)   |
 *              +------------+                |
 *              |  64 bytes  |                |
 *              |  for rwo   | <- SP (temp)   |
 *              +============+                ^
 */

/* The size of the call stub's frame (must be a multiple of 8).
 */
#define CALLSTUB_SZ        8+64

/* offsets w.r.t. the stub frame's pointer */
#define CALLSTUB_NODEID (-4)
#define CALLSTUB_RETFP RETMSG_FP

#endif /* !_CALL_STUB_ */








