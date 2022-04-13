#include "../list.h"

void operation(list_t list) {
  int i = 0;
  while (list) {
    if (i % 2 == 0) list->data += 1;
    else list->data -= 1;
    list = list->next; i++;
  }
}
