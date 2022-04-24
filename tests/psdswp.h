#ifndef PAR_SUPPORT_H_
#define PAR_SUPPORT_H_

#include <stdarg.h>
#include <stdlib.h>
#include <pthread.h>

// LLVM type: i8* (i32, ...)
// Variadic arguments are i32 specifying the number of instances of that array
void* createSyncArrays(int numArrays, ...);
// LLVM type: void (i8*, i32, ...)
// Again, variadic arguments are the number of instances
void freeSyncArrays(void* syncArrays, int numArrays, ...);
// LLVM type: void (i8*, i32, i32, i64)
// Currently, only supports passing scalars values up to 64-bits, they must be
// converted into an i64 by bit-extension and bitcasting (as needed)
void produce(void* syncArrays, int toArray, int toRepl, int64_t value);
// LLVM type: i64 (i8*, i32, i32)
int64_t consume(void* syncArrays, int fromArray, int fromRepl);

// Providing the function definitions in the header means LLVM won't just
// delete the unused declarations

struct syncArrayElem {
  int64_t val;
  struct syncArrayElem* prev;
  struct syncArrayElem* next;
};

struct syncArray {
  struct syncArrayElem header;
  struct syncArrayElem footer;
  pthread_mutex_t lock;
  pthread_cond_t empty;
};

// Marking as optNone to avoid us trying to parallelize these, annoying since
// optnone prevents all optimizations (TODO: find some other way?)
void* createSyncArrays(int numArrays, ...) __attribute__((optnone)) {
  struct syncArray** syncArrays = malloc(sizeof(struct syncArray*) * numArrays);

  va_list args;
  va_start(args, numArrays);

  for (int i = 0; i < numArrays; i++) {
    int instances = va_arg(args, int);
    syncArrays[i] = malloc(sizeof(struct syncArray) * instances);
    // Initialize the syncArrays
    for (int j = 0; j < instances; j++) {
      syncArrays[i][j].header.prev = NULL;
      syncArrays[i][j].header.next = &(syncArrays[i][j].footer);
      syncArrays[i][j].footer.prev = &(syncArrays[i][j].header);
      syncArrays[i][j].footer.next = NULL;

      pthread_mutex_init(&(syncArrays[i][j].lock), NULL);
      pthread_cond_init(&(syncArrays[i][j].empty), NULL);
    }
  }

  va_end(args);
  return syncArrays;
}

void freeSyncArrays(void* syncArrays, int numArrays, ...) __attribute__((optnone)) {
  struct syncArray** pipelines = (struct syncArray**) syncArrays;
  
  va_list args;
  va_start(args, numArrays);

  for (int i = 0; i < numArrays; i++) {
    int instances = va_arg(args, int);
    for (int j = 0; j < instances; j++) {
      pthread_mutex_destroy(&(pipelines[i][j].lock));
      pthread_cond_destroy(&(pipelines[i][j].empty));
      // We don't free any data from the list, because it should all be
      // consumed already
    }
    free(pipelines[i]);
  }
  free(pipelines);
}

void produce(void* syncArrays, int toArray, int toRepl, int64_t value) __attribute__((optnone)) {
  struct syncArray** pipelines = (struct syncArray**) syncArrays;
  struct syncArray* syncArray = &(pipelines[toArray][toRepl]);

  struct syncArrayElem* node = malloc(sizeof(struct syncArrayElem));
  node->val = value;
  node->next = &(syncArray->footer);

  pthread_mutex_lock(&syncArray->lock);

  node->prev = syncArray->footer.prev;
  node->prev->next = node;
  node->next->prev = node;

  pthread_cond_signal(&syncArray->empty);
  pthread_mutex_unlock(&syncArray->lock);
}

int64_t consume(void* syncArrays, int fromArray, int fromRepl) __attribute__((optnone)) {
  struct syncArray** pipelines = (struct syncArray**) syncArrays;
  struct syncArray* syncArray = &(pipelines[fromArray][fromRepl]);

  pthread_mutex_lock(&syncArray->lock);
  while (syncArray->header.next == &(syncArray->footer)) {
    pthread_cond_wait(&syncArray->empty, &syncArray->lock);
  }

  int64_t value = syncArray->header.next->val;
  struct syncArrayElem* elem = syncArray->header.next;
  syncArray->header.next = elem->next;
  elem->next = elem->prev;
  free(elem);

  pthread_mutex_unlock(&syncArray->lock);
  return value;
}

#endif // PAR_SUPPORT_H_
