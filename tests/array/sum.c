#include "../array.h"

void operation(elem_t* array, len_t length) {
  elem_t sum = 0;
  for (len_t i = 0; i < length; i++) {
    sum += array[i];
  }
  array[0] = sum;
}
