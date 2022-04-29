#include "../list.h"

void operation(list_t list) {
  list_t prev = NULL;
  while (list) {
    if (prev && prev->data % 2 == 0) list->data += 1;
    else list->data -= 1;
    prev = list;
    list = list->next;
  }
}
