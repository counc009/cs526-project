#include "../list.h"
#include "../psdswp.h"

void operation(list_t list) {
  while (list) {
    if (list->next && list->next->data % 2 == 0) list->data += 1;
    else list->data -= 1;
    list = list->next;
  }
}
