#include "../array.h"

void operation(elem_t* array, len_t length) {
  for (len_t i = 0; i < length; i++) {
    elem_t elem = array[i];
    elem_t j = 0;
    while (j * j < elem) j++;

    if (j * j != elem) array[i] = elem * elem;
  }
}
