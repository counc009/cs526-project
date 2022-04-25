#include "../list.h"

void operation(list_t list) {
  elem_t sum = 0;
  list_t head = list;
  while (list) {
    sum += list->data;
    list = list->next;
  }
  head->data = sum;
}
