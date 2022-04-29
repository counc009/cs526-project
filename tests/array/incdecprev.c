#include "../array.h"

void operation(elem_t* array, len_t length) {
  for (len_t i = 0; i < length; i++) {
    if (i > 0 && array[i-1] % 2 == 0) array[i] += 1;
    else array[i] -= 1;
  }
}
