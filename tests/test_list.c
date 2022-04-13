#include <stdio.h>
#include <stdlib.h>

#include "list.h"

void printList(list_t list);

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <n> <seed>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);
  if (n <= 0) {
    fprintf(stderr, "<n> must be positive\n");
    exit(1);
  }

  unsigned seed = atoi(argv[2]);
  if (seed == 0) {
    fprintf(stderr, "<seed> must be non-zero\n");
    exit(1);
  }
  srand(seed);

  list_t head = malloc(sizeof(struct node));
  head->data = rand() % n;
  head->next = NULL;

  list_t curr = head;
  for (int i = 1; i < n; i++) {
    list_t node = malloc(sizeof(struct node));
    node->data = rand() % n;
    node->next = NULL;
    curr->next = node;
    curr = node;
  }

  printf("Before: "); printList(head);
  operation(head);
  printf("After:  "); printList(head);
  
  curr = head;
  while (curr) {
    list_t tmp = curr;
    curr = curr->next;
    free(tmp);
  }
  return 0;
}

void printList(list_t list) {
  while (list) {
    printf("%2d ", list->data);
    list = list->next;
  }
  printf("\n");
}
