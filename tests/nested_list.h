typedef int elem_t;
struct node {
  elem_t data;
  struct node* restrict next;
};
struct lnode {
  struct node* restrict data;
  struct lnode* restrict next;
};
typedef struct lnode* restrict list_t;

void operation(list_t list);

#ifndef NULL
#define NULL 0L
#endif
