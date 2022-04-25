#include "psdswp.h"

#include <stdarg.h>
#include <stdlib.h>
#include <pthread.h>

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

void* createSyncArrays(int numArrays, ...) {
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

void freeSyncArrays(void* syncArrays, int numArrays, ...) {
  struct syncArray** pipelines = (struct syncArray**) syncArrays;
  
  va_list args;
  va_start(args, numArrays);

  for (int i = 0; i < numArrays; i++) {
    int instances = va_arg(args, int);
    for (int j = 0; j < instances; j++) {
      pthread_mutex_destroy(&(pipelines[i][j].lock));
      pthread_cond_destroy(&(pipelines[i][j].empty));
      // We may free one element as this off-by-one occurs with the loop
      // condition
      if (pipelines[i][j].header.next != &(pipelines[i][j].footer)) {
        free(pipelines[i][j].header.next);
      }
    }
    free(pipelines[i]);
  }
  free(pipelines);
}

void produce(void* syncArrays, int toArray, int toRepl, int64_t value) {
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

int64_t consume(void* syncArrays, int fromArray, int fromRepl) {
  struct syncArray** pipelines = (struct syncArray**) syncArrays;
  struct syncArray* syncArray = &(pipelines[fromArray][fromRepl]);

  pthread_mutex_lock(&syncArray->lock);
  while (syncArray->header.next == &(syncArray->footer)) {
    pthread_cond_wait(&syncArray->empty, &syncArray->lock);
  }

  int64_t value = syncArray->header.next->val;
  struct syncArrayElem* elem = syncArray->header.next;
  syncArray->header.next = elem->next;
  elem->next->prev = elem->prev;
  free(elem);

  pthread_mutex_unlock(&syncArray->lock);
  return value;
}

pthread_t launchStage(void* argument, void*(*func)(void*)) {
  pthread_t thread;
  if (pthread_create(&thread, NULL, func, argument)) {
    exit(1);
  }
  return thread;
}

void waitForStage(pthread_t thread) {
  if (pthread_join(thread, NULL)) {
    exit(1);
  }
}
