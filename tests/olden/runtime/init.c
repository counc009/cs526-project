/* $Log: init.c,v $
 * Revision 1.1  1995/09/22  17:34:36  mcc
 * Initial revision
 *
 * Revision 1.1  1995/09/13  18:59:52  mcc
 * Initial revision
 *
 *
 */

/* init.c
 *
 */

#include <cm/cmmd.h>
#include <fcntl.h>
#include "constants.h"
#include "cache.h"
#include "msg.h"

int     __NumNodes;
int     __NDim;
unsigned int	__MyNodeId;          /* node id */

void Stats()
{
  if (CMMD_self_address() < __NumNodes) 
    {
#ifdef CHIP_STATS
      PrintChipStats();
#endif
#ifdef CACHE_STATS
      PrintCacheStats();
#endif      
#ifdef PROF
      OutputProfStats();
#endif
#ifdef VISUALIZE
      visCleanUp();
#endif
    }
  
}

/* SPMDInit:
 *
 */
int SPMDInit (argc,argv)
     int argc;
     char *argv[];
{
    int		retval;

    CMMD_fset_io_mode(stdout, CMMD_independent);
    fcntl(fileno(stdout), F_SETFL, O_APPEND);
    __InitCache2();
    __InitChips();
    __InitMsg();
    
    __MyNodeId = CMMD_self_address();
    __InitRegs(__MyNodeId);
    __NumNodes = CMMD_partition_size();
    retval=dealwithargs(argc,argv);
#ifdef VISUALIZE
    if (__MyNodeId < __NumNodes) visInit(__MyNodeId,PG_TBL_SZ);
#endif
    CMMD_sync_with_nodes();
    if (__MyNodeId >= __NumNodes) exit(0);
    if (__MyNodeId != 0) __MsgDispatch(__getsp(),0);
    return retval;
} /* end of SPMDInit */

/* __ShutDown: Should be called only by processor 0
 */
void __ShutDown (result)
  int result;
{
  int worker;
  int size = CMMD_partition_size(); 
  int me = __MyNodeId;
  
  size = (size > MAXPROC) ? MAXPROC : size;

  if (size > __NumNodes) size = __NumNodes;
  for (worker=0; worker < size; worker++)
    if (worker != me) 
      CMAML_request(worker,__recv_msg,0,0,STOP_MSG,me);
  Stats();
  exit(0);
} /* __Shutdown */




