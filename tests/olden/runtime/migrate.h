/* $Log: migrate.h,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 *
 */

#ifndef MIGRATE_INCLUDED
#define MIGRATE_INCLUDED

#include "constants.h"

/* A "migrate on reference" message has the following layout:
 *
 *       0:     +------------+ <-- msgptr
 *              |  RWO       |
 *              |  space     |
 *              |  64 bytes  |
 *              +------------+
 *      64:     |  splink    |
 *              +------------|
 *      68:     |            |
 *              |  locals    |
 *              |            |
 *              +------------+
 *              |            |
 *              |  cs buf    |
 *              |            |
 *              +------------+
 *      -8:     |  ret addr  |
 *              +------------+
 *      -4:     | fp link    |
 *              +------------+ <-- fp
 *        0:    |  nodeId    | 
 *              +------------+
 *        4:    |  retPC     |
 *              +------------+
 *        8:    |  savedFP   |
 *              +------------+
 *       12:    |  p         |
 *              +------------+
 *       16:    |  retsize   |
 *              +------------+
 *       20:    |  frameSzB  |
 *              +------------+
 *       24:    |  argsize   |
 *              +------------+
 *       28:    |  modmask   |
 *              +------------+
 *       32:    |  modmask   |
 *              +------------+
 */

#ifndef _ASM_
typedef struct {
  word_t nodeId; /* inital nodeId of function */
  word_t returnPC; /* where to begin execution */
  word_t savedFP; /* initial FP of function */
  word_t p; /* non-local pointer */
  word_t retsize; /* size of return value */
  word_t frameSzB; /* size of frame */
  word_t argsize; /* size of argbuild area */
  word_t modmask[2];
} refmsg_t;

#define REFMSG_SZ(n)	(sizeof(refmsg_t)+(((n)-1)*sizeof(word_t)))

extern void __CallMsgRecv();
extern void __ReturnMsgRecv();
#endif

#define CALLMSG_NODEID 0
#define CALLMSG_PC 4
#define CALLMSG_FP 8
#define CALLMSG_P 12
#define CALLMSG_RETSIZE 16
#define CALLMSG_FRAMESIZE 20
#define CALLMSG_ARGSIZE 24

/* A "migrate on return" message has the following layout:
 *
 *		+------------------+
 *	0:	|       fp         |
 *		+------------------+
 *	4:	|                  |
 *		|     i0-i5        |
 *		|                  |
 *		+------------------+
 *	28:	| return val (r16) |
 *		+------------------+
 *	32:	|   returnSize     |
 *		+------------------+
 *      36:     |   framesize      |
 *		+------------------+
 *      40:     |   argsize        |
 *		+------------------+
 *      44:     |   mask[0]        |
 *		+------------------+
 *      48:     |   mask[1]        |
 *		+------------------+
 *      52:     |   nextmsg        |
 *		+------------------+
 *      56:     |   returnfl[2]    |
 *		+------------------+
 *      64:     |   retstruct      |
 *              |   [STRUCTSIZE]   |
 *              |                  |
 *		+------------------+
 */
#ifndef _ASM_
#define STRUCTSIZE  60
typedef struct {	    /* Migrate on return messages */
    word_t	fp;	/* fp of the thread that is migrating */
    word_t	locs[6];	/* the callee-save registers (from i0-i5) */
    word_t	returnVal;	/* the return value (for o0) */
    word_t	retsize;
    word_t      framesz;
    word_t      argsize;
    word_t      modmask[2];
    word_t      next_retmsg; /* used in keeping freelist */
    word_t      flreturn[2];
    word_t      retstruct[STRUCTSIZE];
  } retmsg_t;

#define BASE_RETMSG_SZ 128 /* used only for memcpy */

#define NUMRETBUFS 300

extern int __FreeRetMsg;

typedef union {
    double align;
    retmsg_t retmsg;
  } aligned_retmsg;
extern aligned_retmsg __retmsgs[NUMRETBUFS];
#endif

#define RETMSG_SIZE 320  /* Must be 16 byte aligned */

/** Assembly offsets **/
#define RETMSG_FP               0
#define RETMSG_i0 		4
#define RETMSG_i1		8
#define RETMSG_i2		12
#define RETMSG_i3		16
#define RETMSG_i4		20
#define RETMSG_i5		24
#define RETMSG_o0               28
#define RETMSG_RETSIZE          32
#define RETMSG_FS               36
#define RETMSG_ARGSIZE          40
#define RETMSG_FLRETURN         56
#define RETMSG_RETSTRUCT        64

/* maximum size of migration message */
#define MSG_BUF_SZ	 8192

#ifndef _ASM_
typedef union {
  char	    buf[MSG_BUF_SZ];
  refmsg_t    ref;
  retmsg_t    ret;
  double align_me;
} msg_t;
#endif


#endif
