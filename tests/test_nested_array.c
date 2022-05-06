#include <stdio.h>
#include <stdlib.h>

#include "nested_array.h"

void printArray(array_t* array, len_t len1, len_t len2);

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <n> <m> <seed>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);
  if (n <= 0) {
    fprintf(stderr, "<n> must be positive\n");
    exit(1);
  }

  int m = atoi(argv[2]);
  if (m <= 0) {
    fprintf(stderr, "<m> must be positive\n");
    exit(1);
  }

  unsigned seed = atoi(argv[3]);
  if (seed == 0) {
    fprintf(stderr, "<seed> must be non-zero\n");
    exit(1);
  }
  srand(seed);

  array_t* array = malloc(sizeof(array_t) * n);
  for (int i = 0; i < n; i++) {
    array[i] = malloc(sizeof(elem_t) * m);
    for (int j = 0; j < m; j++) {
      array[i][j] = rand() % (n * m);
    }
  }

  printf("Before: "); printArray(array, n, m);
  operation(array, n, m);
  printf("After:  "); printArray(array, n, m);
  
  for (int i = 0; i < n; i++) free(array[i]);
  free(array);
  return 0;
}

void printArray(array_t* array, len_t len1, len_t len2) {
  for (len_t i = 0; i < len1; i++) {
    for (len_t j = 0; j < len2; j++) {
      printf("%2d ", array[i][j]);
    }
    printf("| ");
  }
  printf("\n");
}
