'''
These functions build a decision tree, which is output as a table 
contained in a numpy array.

The tree does single-output, binary classification using the Gini impurity 
criterion.

author: David Thaler
date: August 2017
'''
import numpy as np
import numba
import gini_splitter
import mse_splitter
import tree_constants as tc


def build_tree(x, y, max_features=-1, min_leaf=1, min_split=2, max_depth=-1, 
                depth=0, node_num=0, criterion='gini'):
    '''
    Recursively build a decision tree. 
    Returns a 2-D array of shape (num_nodes x 7) that describes the tree.
    Each row represents a node in pre-order (root, left, right).
    See the module comment for the column definitions.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        max_features: try up to this number of features per split
            default of -1 for all features
        min_leaf: each branch must have at least min_leaf samples
            default 1
        min_split: do not split node if it has less than `min_split` samples
        max_depth: stop at this depth; default of -1 for no depth limit
        depth: the depth of this node; the root is 0
        node_num: the node number of this node
            default 0 is for the root node
        criterion: either 'gini' for classification or 'mse' for regression
            default is 'gini'

    Returns:
        2-D numpy array of dtype 'float' with 
    '''
    if criterion == 'gini':
        split = gini_splitter.split
    elif criterion == 'mse':
        split = mse_splitter.split
    else:
        raise ValueError("'criterion' must be one of 'gini' or 'mse'")
    ct = len(y)
    val = y.sum() / ct
    if (ct < min_split) or (depth == max_depth):
        return np.array([[tc.NO_FEATURE, tc.NO_THR, node_num,
                         tc.NO_CHILD, tc.NO_CHILD, ct, val]])
    feature, thr = split(x, y, max_features=max_features, min_leaf=min_leaf)
    if feature == tc.NO_FEATURE:
        return np.array([[feature, thr, node_num, tc.NO_CHILD, tc.NO_CHILD, ct, val]])
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], max_features, min_leaf, 
                            min_split, max_depth, depth + 1, left_root, criterion)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], max_features, min_leaf, 
                            min_split, max_depth, depth + 1, right_root, criterion)
    root = np.array([[feature, thr, node_num, left_root, right_root, ct, val]])
    return np.concatenate([root, left_tree, right_tree])


@numba.jit
def apply(tree, x):
    '''
    Finds the node number in the provided tree (from build_tree) that each
    instance in x lands in.

    NB: using numba.jit w/o nopython=True option allows astype(int) at end

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype int) of leaf node numbers for each point in x.
    '''
    out = np.zeros(len(x))
    for k in range(len(x)):
        node = 0                                           # the root
        while tree[node, tc.FEATURE_COL] >= 0:             # not a leaf
            feature_num = int(tree[node, tc.FEATURE_COL])
            thr = tree[node, tc.THR_COL]
            if x[k, feature_num] <= thr:
                node = int(tree[node, tc.CHILD_LEFT_COL])
            else:
                node = int(tree[node, tc.CHILD_RIGHT_COL])
        out[k] = node
    return out.astype(int)


def prediction_value(tree, x):
    '''
    Returns the prediction value for each row in x given a tree.
    For regression trees, this is the estimate. For classification
    trees, it is p(y=1|x).

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype float) of prediction values
    '''
    leaf_idx = apply(tree, x)
    return tree[leaf_idx, tc.VAL_COL]
