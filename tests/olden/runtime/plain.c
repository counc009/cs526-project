/* $Log: plain.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

#include "constants.h"
#include "msg.h"
#include <cm/cmmd.h>
#include <fcntl.h>
unsigned int __MyNodeId;
int __NumNodes;
int __NDim;
int Misses,ConflictMisses;

int  SPMDInit(argc,argv)
  int argc;
  char *argv[];
{
    int     retval;

    CMMD_fset_io_mode(stdout, CMMD_independent);
    fcntl(fileno(stdout), F_SETFL, O_APPEND);
    __MyNodeId = CMMD_self_address();
    __InitRegs(__MyNodeId);
    __NumNodes = 4 ; /*CMMD_partition_size();*/
    __NDim = 2;
    retval=dealwithargs(argc,argv);
    if (__MyNodeId >= MAXPROC) exit(0);
    if (__MyNodeId != 0) exit(0);
    return retval;
}  

void chatting (fmt, a, b, c, d, e, f, g)
    char *fmt;
    int     a, b, c, d, e, f, g;
{
    char buf[256];

    sprintf(buf, "[%d] ", CMMD_self_address());
    sprintf(buf+strlen(buf), fmt, a, b, c, d, e, f, g);
    printf("%s", buf);
}

void __ShutDown()
{
exit(0);
}
