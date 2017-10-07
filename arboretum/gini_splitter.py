'''
The split function here finds splits that minimize gini impurity.
It runs under numba for speed, since these are the innermost loops in 
decision tree fitting.

author: David Thaler
date: August 2017
'''
import numpy as np
import numba
from . import tree_constants as tc

@numba.jit(nopython=True)
def split(x, y,  wts, max_features=-1, min_leaf=-1):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in Gini impurity.
    Each side of the split must have at least min_leaf samples.

    Note:
        If no valid split is found after max_features, and max_features is less
        than the number of features, splitter will continue to try features, one
        at a time, in random order, until a valid split is found, or until all 
        features have been tried.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        wts: sample weights, use ones for unweighted case
        max_features: try up to this number of features per split
            default of -1 for all features
        min_leaf: min sample weight for a leaf
            default of -1 for wts.min()

    Returns:
        2-tuple of feature index and split threshold of best split.
    '''
    m, n = x.shape
    NO_SPLIT = (tc.NO_FEATURE, tc.NO_THR, 0.)
    improve = False
    if min_leaf == -1:              # not set
        min_leaf = wts.min()        # wts is ones if unweighted
    if max_features == -1:
        max_features = n
    tot_wt = wts.sum()
    ywt = y * wts
    tot_ywt = ywt.sum()

    # the Gini impurity of this node before splitting
    node_score = 1 - (tot_ywt/tot_wt)**2 - ((tot_wt - tot_ywt)/tot_wt)**2

    # a code optimization for pure nodes
    if node_score==0:
        return NO_SPLIT
    col_order = np.random.choice(np.arange(n), size=n, replace=False)
    
    # Stores score, threshold for each feature (1 > max value for gini)
    results = np.ones((n, 2))
    
    for col_ct in range(n):
        if col_ct >= max_features and improve:
            break
        feature_idx = col_order[col_ct]
        f = x[:, feature_idx]

        # Produce 3 arrays:
        # 1) sorted unique values in f
        # 2) count of each unique value (usually 1)
        # 3) # of positives for each unique value
        ntot = np.zeros(m)
        uniq = np.zeros(m)
        npos = np.zeros(m)
        cur_val = np.nan
        num_uniq = 0
        for idx in np.argsort(f):
            if f[idx] != cur_val:
                cur_val = f[idx]
                uniq[num_uniq] = cur_val
                num_uniq += 1
            ntot[num_uniq - 1] += wts[idx]
            npos[num_uniq - 1] += ywt[idx]
        uniq = uniq[:num_uniq]
        npos = npos[:num_uniq]
        ntot = ntot[:num_uniq]

        # Get cumulative counts/positives/negatives for each possible split
        nleft = ntot.cumsum()
        nright = tot_wt - nleft
        npos_left = npos.cumsum()
        nneg_left = nleft - npos_left
        npos_right = tot_ywt - npos_left
        nneg_right = nright - npos_right

        # trim to valid splits (at least min_leaf both sides)
        mask = (nleft >= min_leaf) & (nright >= min_leaf)
        # There must be at least one value on each side, 
        # so the last position is not valid, 
        # as there would be an empty right branch
        mask[-1] = False

        # at this point there might be no valid splits
        if not mask.any():
            continue

        # we need this to set the split index w/o a search
        a = mask.argmax()
        b = -mask[::-1].argmax()
        
        nleft = nleft[a:b]
        nright = nright[a:b]
        npos_left = npos_left[a:b]
        npos_right = npos_right[a:b]
        nneg_left = nneg_left[a:b]
        nneg_right = nneg_right[a:b]

        # This is a gini proxy from the form 2 * (p1 * p2)
        gini_split = (npos_left * nneg_left / nleft) + (npos_right * nneg_right / nright)

        # Select the best split
        split_pos = gini_split.argmin()
        # gini_split holds a proxy score that differs from gini by (2/tot_wt)
        split_score = (2/tot_wt) * gini_split[split_pos]
        split_idx = a + split_pos
        thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
        results[feature_idx] = (split_score, thr)
        if split_score < node_score:
            improve = True
    best_split_idx = results[:, 0].argmin()
    best_score = results[best_split_idx, 0]
    if best_score < node_score:
        best_thr = results[best_split_idx, 1]
        return (best_split_idx, best_thr, node_score - best_score)
    else:
        return NO_SPLIT
    return results
