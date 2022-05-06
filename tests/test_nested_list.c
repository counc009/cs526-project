#include <stdio.h>
#include <stdlib.h>

#include "nested_list.h"

void printList(list_t list);

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <n> <m> <seed>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);
  if (n <= 0) {
    fprintf(stderr, "<n> must be positive\n");
    exit(1);
  }

  int m = atoi(argv[2]);
  if (n <= 0) {
    fprintf(stderr, "<m> must be positive\n");
    exit(1);
  }

  unsigned seed = atoi(argv[3]);
  if (seed == 0) {
    fprintf(stderr, "<seed> must be non-zero\n");
    exit(1);
  }
  srand(seed);

  list_t head = NULL;
  list_t curr = head;
  for (int i = 0; i < n; i++) {
    list_t node = malloc(sizeof(struct lnode));
    node->next = NULL;

    node->data = malloc(sizeof(struct node));
    node->data->data = rand() % (n * m);
    node->data->next = NULL;

    struct node* cnode = node->data;
    for (int j = 1; j < m; j++) {
      struct node* nd = malloc(sizeof(struct node));
      nd->data = rand() % (n * m);
      nd->next = NULL;
      cnode->next = nd;
      cnode = nd;
    }

    if (curr) curr->next = node;
    else head = node;
    curr = node;
  }

  printf("Before: "); printList(head);
  operation(head);
  printf("After:  "); printList(head);
  
  curr = head;
  while (curr) {
    list_t tmp = curr;
    struct node* cn = tmp->data;
    while (cn) {
      struct node* t = cn;
      cn = cn->next;
      free(t);
    }
    curr = curr->next;
    free(tmp);
  }
  return 0;
}

void printList(list_t list) {
  while (list) {
    struct node* ptr = list->data;
    while (ptr) {
      printf("%2d ", ptr->data);
      ptr = ptr->next;
    }
    printf("| ");
    list = list->next;
  }
  printf("\n");
}
