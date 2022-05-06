#ifndef __VIS_H_
#define __VIS_H_

FILE *visFile;

int visInit(int nodeid, int pages);
void visCleanUp(void);

void visProc(void);
void visProcMigIn(void);
void visProcMigOut(void);
void visProcMigRet(void);
void visProcSteal(void);
void visProcTouch(void);

void visCacheHit(int page);
void visCacheMiss(int page);
void visCacheRemoteWrite(void);

void visStackAlloc(unsigned int loc, int size);
void visStackDealloc(unsigned int loc);
void visStackDeactivate(unsigned int loc);

void visFstackPush(void);
void visFstackPop(void);

#endif
