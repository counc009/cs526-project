#include "../nested_array.h"

void operation(array_t* restrict array, len_t len1, len_t len2) {
  for (len_t i = 0; i < len1; i++) {
    for (len_t j = 0; j < len2; j++) {
      array[i][j] += 1;
    }
  }
}
