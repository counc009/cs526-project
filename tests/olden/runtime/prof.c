#include "runtime.h"
#include "constants.h"
#include "stdio.h"

#define ISLOCAL(p) ((PID(p)==__MyNodeId) || (p==NULL))
#define LI_TBL_SZ 1000

struct LineInfo {
  struct LineInfo *next;
  char *filename;
  int lineno;
  int access_type;
  union {
    struct {
      int locals;
      int hits;
      int misses;
      int rem_writes;
    } cache; 
    struct {
      int locals;
      int migrations;
    } migrate;
  } u;
};

static struct LineInfo *LITable[LI_TBL_SZ];

static struct LineInfo *FindOrInsert(filename,lineno,access_type)
  char *filename;
  int lineno;
  int access_type;
{
  struct LineInfo *retval;
  int hash = ((unsigned int) filename + lineno) % LI_TBL_SZ;

  if (access_type==REMOTE_WRITE_ACCESS) access_type = CACHE_ACCESS;

  for (retval = LITable[hash]; retval != NULL; retval = retval->next)
    {
      if (retval->lineno==lineno && retval->filename == filename &&
          retval->access_type == access_type)
        return retval;
    }
  retval = malloc(sizeof(*retval));
  retval->lineno = lineno;
  retval->filename = filename;
  retval->access_type = access_type;
  retval->u.cache.locals = 0;
  retval->u.cache.hits = 0;
  retval->u.cache.misses = 0;
  retval->u.cache.rem_writes = 0;
  retval->next = LITable[hash];
  LITable[hash] = retval; 
  return retval;
}

void OutputProfStats()
{
  FILE *fp;
  char filename[100];
  int i;
  struct LineInfo *lis;
  
  sprintf(filename,"prof.%d.pn%02d",getpid(),__MyNodeId);
  fp = fopen(filename,"w");
  for (i=0; i<LI_TBL_SZ; i++) {
    for(lis = LITable[i]; lis; lis=lis->next) {
      if (lis->access_type == CACHE_ACCESS)
        fprintf(fp,"File: %s Line: %05d Loc: %d Hit: %d Miss: %d Rem_Wr: %d\n",
              lis->filename,lis->lineno,
              lis->u.cache.locals,lis->u.cache.hits,lis->u.cache.misses,
              lis->u.cache.rem_writes);
      else
        fprintf(fp,"File: %s Line: %05d Local: %d Migrates: %d\n",
              lis->filename,lis->lineno,
              lis->u.migrate.locals,lis->u.migrate.migrations);
    } 
  }
  fclose(fp);
}

void __prof2(access_type,p,filename,lineno) 
  int access_type;
  unsigned int p;
  char *filename;
  int lineno;
{
  struct LineInfo *lis;

  lis = FindOrInsert(filename,lineno,access_type);
  if (access_type==CACHE_ACCESS) {
    if (ISLOCAL(p)) {
      lis->u.cache.locals++;
      }
    else {
      if (__Cache2Lookup(p))
        lis->u.cache.hits++;
      else
        lis->u.cache.misses++;
      }
    }
  else if (access_type==MIG_ACCESS) {
    if (ISLOCAL(p)) {
      lis->u.migrate.locals++;
      }
    else {
      lis->u.migrate.migrations++;
      }
    }
  else {
    if (ISLOCAL(p)) {
      lis->u.cache.locals++;
      }
    else {
      lis->u.cache.rem_writes++;
      if (__Cache2Lookup(p))
        lis->u.cache.hits++;
      else
        lis->u.cache.misses++;
      }
  }
}
