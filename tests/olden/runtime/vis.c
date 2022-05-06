/*  
 *  Jessica Weinstein
 *  Junior Independent Work
 *  Spring '96
 */
#include <cm/cmmd.h>
#include <fcntl.h>

#include <stdio.h>
#include "vis.h"
double CMMD_node_timer_elapsed();

FILE *visFile;

int visInit(int nodeid, int pages)
{
    char name[80];

    sprintf(name, "olden.trace%d", nodeid);
    visFile = fopen(name, "w");
    if(!visFile) return -1;
    CMMD_node_timer_clear(31);
    CMMD_node_timer_start(31);
    CMMD_fset_io_mode(visFile, CMMD_independent);
    fprintf(visFile, "Cache %d\n", pages);
    return 0;
} 

void visProc(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Proc %f\n", time);
}

void visProcMigIn(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "MigIn %f\n", time);
}

void visProcMigOut(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "MigOut %f\n", time);
}

void visProcMigRet(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "MigRet %f\n", time);
}

void visProcSteal(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Steal %f\n", time);
}

void visProcTouch(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Touch %f\n", time);
}

void visCacheHit(int page)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Hit %f %d\n", time, page);
}

void visCacheMiss(int page)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Miss %f %d\n", time, page);
}

void visCacheRemoteWrite(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "RemWrite %f\n", time);
}

void visStackAlloc(unsigned int loc, int size)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Alloc %f %u %u\n", time, loc, size);
}

void visStackDealloc(unsigned int loc)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Dealloc %f %u\n", time, loc);
}

void visStackDeactivate(unsigned int loc)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Deactivate %f %u\n", time, loc);
}

void visFstackPush(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Push %f\n", time);
}

void visFstackPop(void)
{
    double time;
    CMMD_node_timer_stop(31);
    time = CMMD_node_timer_elapsed(31);
    CMMD_node_timer_start(31);
    fprintf(visFile, "Pop %f\n", time);
}

void visCleanUp(void)
{
    CMMD_node_timer_stop(31);
    fclose(visFile);
}
