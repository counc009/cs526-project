#include <stdio.h>
#include <stdlib.h>

struct node {
  int val;
  struct node* next;
};

struct node* generateList(unsigned seed) {
  srand(seed);
  
  struct node* list = malloc(sizeof(struct node));
  list->val = -100;
  list->next = NULL;

  struct node* ptr = list;

  while (rand() % 10 != 0) {
    struct node* next = malloc(sizeof(struct node));
    next->val = rand() % 10;
    next->next = NULL;
    ptr->next = next;
    ptr = next;
  }

  return list;
}

void printList(struct node* list) {
  struct node* ptr = list;
  while ((ptr = ptr->next)) {
    printf("%d\n", ptr->val);
  }
}

void incList(struct node* list) {
  struct node* ptr = list;
  while ((ptr = ptr->next)) {
    ptr->val = ptr->val + 1;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <seed>\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  struct node* list = generateList(n);

  printList(list);
  printf("================================================================\n");
  incList(list);
  printList(list);

  return 0;
}
