#include "../list.h"

void operation(list_t list) {
  while (list) {
    list->data += 1;
    list = list->next;
  }
}
