#include "../nested_list.h"

void operation(list_t list) {
  while (list) {
    struct node* restrict data = list->data;
    while (data) {
      data->data += 1;
      data = data->next;
    }
    list = list->next;
  }
}
