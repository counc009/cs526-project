#include "../nested_array.h"

void operation(restrict array_t* array, len_t len1, len_t len2) {
  for (len_t i = 0; i < len1; i++) {
    for (len_t j = 0; j < len2; j++) {
      elem_t elem = array[i][j];
      elem_t k = 0;
      while (k * k < elem) k++;

      if (k * k != elem) array[i][j] = elem * elem;
    }
  }
}
