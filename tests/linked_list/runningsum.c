#include "../list.h"

void operation(list_t list) {
  elem_t sum = 0;
  while (list) {
    sum += list->data;
    list->data = sum;
    list = list->next;
  }
}
