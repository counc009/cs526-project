#include "../tree.h"
#include "../psdswp.h"

void operation(tree_t tree) {
  while (tree) {
    elem_t elem = tree->data;
    elem_t j = 0;
    while (j * j < elem) j++;

    if (j * j != elem) tree->data = elem * elem;

    if (tree->left) tree = tree->left;
    else if (tree->right) tree = tree->right;
    else if (tree->parent == NULL) tree = NULL;
    else if (tree->parent->left == tree) {
      while (tree->parent && (tree->parent->right == NULL
                           || tree->parent->right == tree)) {
        tree = tree->parent;
      }
      if (tree->parent == NULL) tree = NULL;
      else tree = tree->parent->right;
    } else {
      do {
        tree = tree->parent;
      } while (tree && tree->parent && tree->parent->right == tree);
      if (tree && tree->parent) tree = tree->parent->right;
      else tree = NULL;
    }
  }
}
