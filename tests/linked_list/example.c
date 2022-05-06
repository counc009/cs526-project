#include "../list.h"

// This file contains the running example used in our report

void operation(list_t list) {
  elem_t sum = 0;
  list_t head = list;
  while (list) {
    list->data += 1;
    sum += list->data;
    list = list->next;
  }
  head->data = sum;
}
