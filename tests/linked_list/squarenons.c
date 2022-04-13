#include "../list.h"

void operation(list_t list) {
  while (list) {
    elem_t elem = list->data;
    elem_t j = 0;
    while (j * j < elem) j++;

    if (j * j != elem) list->data = elem * elem;
    list = list->next;
  }
}
