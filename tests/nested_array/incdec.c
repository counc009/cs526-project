#include "../nested_array.h"

void operation(restrict array_t* array, len_t len1, len_t len2) {
  for (len_t i = 0; i < len1; i++) {
    for (len_t j = 0; j < len2; j++) {
      if ((i + j) % 2 == 0) array[i][j] += 1;
      else array[i][j] -= 1;
    }
  }
}
