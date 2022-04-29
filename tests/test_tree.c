#include <stdio.h>
#include <stdlib.h>

#include "tree.h"

void printTree(tree_t tree);
tree_t constructTree(int n);
void freeTree(tree_t tree);

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

  tree_t tree = constructTree(n);

  printf("Before: "); printTree(tree); printf("\n");
  operation(tree);
  printf("After:  "); printTree(tree); printf("\n");
  
  freeTree(tree);
  return 0;
}

void printTree(tree_t tree) {
  if (!tree) return;
  printf("%2d L-> ", tree->data);
  printTree(tree->left);
  printf("R-> ");
  printTree(tree->right);
}

tree_t constructHelper(int n, int mod) {
  if (n <= 0) return NULL;

  tree_t root = malloc(sizeof(struct node));
  root->data = rand() % mod;

  int nL = (n-1) / 2;
  int nR = (n-1) - nL;
  root->left = constructHelper(nL, mod);
  if (root->left) root->left->parent = root;
  root->right = constructHelper(nR, mod);
  if (root->right) root->right->parent = root;

  return root;
}

tree_t constructTree(int n) {
  tree_t result = constructHelper(n, n);
  result->parent = NULL;
  return result;
}

void freeTree(tree_t tree) {
  if (!tree) return;
  freeTree(tree->left);
  freeTree(tree->right);
  free(tree);
}
