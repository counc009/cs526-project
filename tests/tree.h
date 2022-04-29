typedef int elem_t;
// These restricts may not actually be correct
struct node {
  elem_t data;
  struct node* restrict parent;
  struct node* restrict left;
  struct node* restrict right;
};
typedef struct node* restrict tree_t;

void operation(tree_t tree);

#ifndef NULL
#define NULL 0L
#endif
