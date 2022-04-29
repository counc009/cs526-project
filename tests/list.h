typedef int elem_t;
struct node {
  elem_t data;
  struct node* restrict next;
};
typedef struct node* restrict list_t;

void operation(list_t list);

#ifndef NULL
#define NULL 0L
#endif
