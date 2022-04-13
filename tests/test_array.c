#include <stdio.h>
#include <stdlib.h>

#include "array.h"

void printArray(elem_t* array, len_t len);

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <n> <seed>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);
  if (n <= 0) {
    fprintf(stderr, "<n> must be positive\n");
    exit(1);
  }

  unsigned seed = atoi(argv[2]);
  if (seed == 0) {
    fprintf(stderr, "<seed> must be non-zero\n");
    exit(1);
  }
  srand(seed);

  elem_t* array = malloc(sizeof(elem_t) * n);
  for (int i = 0; i < n; i++) {
    array[i] = rand() % n;
  }

  printf("Before: "); printArray(array, n);
  operation(array, n);
  printf("After:  "); printArray(array, n);
  
  free(array);
  return 0;
}

void printArray(elem_t* array, len_t len) {
  for (len_t i = 0; i < len; i++) {
    printf("%2d ", array[i]);
  }
  printf("\n");
}
