#include "../tree.h"

void operation(tree_t tree) {
  int depth = 0;
  while (tree) {
    if (depth % 2 == 0) tree->data += 1;
    else tree->data -= 1;

    if (tree->left) { tree = tree->left; depth++; }
    else if (tree->right) { tree = tree->right; depth++; }
    else if (tree->parent == NULL) tree = NULL;
    else if (tree->parent->left == tree) {
      while (tree->parent && (tree->parent->right == NULL
                           || tree->parent->right == tree)) {
        tree = tree->parent;
        depth--;
      }
      if (tree && tree->parent) tree = tree->parent->right; // Maintains the same depth
      else tree = NULL;
    } else {
      do {
        tree = tree->parent;
        depth--;
      } while (tree && tree->parent && tree->parent->right == tree);
      if (tree && tree->parent) tree = tree->parent->right; // Maintains the same depth
      else tree = NULL;
    }
  }
}
