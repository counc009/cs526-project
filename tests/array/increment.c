#include "../array.h"
#include "../psdswp.h"

void operation(elem_t* array, len_t length) {
  for (len_t i = 0; i < length; i++) {
    array[i] += 1;
  }
}
