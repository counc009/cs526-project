#include "../nested_array.h"

void operation(restrict array_t* array, len_t len1, len_t len2) {
  elem_t sum = 0;
  for (len_t i = 0; i < len1; i++) {
    for (len_t j = 0; j < len2; j++) {
      sum += array[i][j];
      array[i][j] = sum;
    }
  }
}
