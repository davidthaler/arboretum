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
    best_feature = tc.NO_FEATURE
    best_thr = tc.NO_THR
    improve = False
    if min_leaf == -1:              # not set
        min_leaf = wts.min()        # wts is ones if unweighted
    if max_features == -1:
        max_features = n
    tot_wt = wts.sum()
    ywt = y * wts
    # the Gini impurity of this node before splitting
    best_score = 1 - (ywt.sum()/tot_wt)**2 - ((tot_wt - ywt.sum())/tot_wt)**2
    # a code optimization for pure nodes
    if (y==y[0]).all():
        return (best_feature, best_thr)
    col_order = np.random.choice(np.arange(n), size=n, replace=False)
    for col_ct in range(n):
        if col_ct >= max_features and improve:
            break
        feature_idx = col_order[col_ct]
        f = x[:, feature_idx]

        # Produce 3 arrays:
        # 1) sorted unique values in f
        # 2) count of each unique value (usually 1)
        # 3) # of positives for each unique value
        sort_idx = f.argsort()
        fsort = f[sort_idx]
        ysort = y[sort_idx]
        wsort = wts[sort_idx]
        ntot = np.zeros(m)
        uniq = np.zeros(m)
        npos = np.zeros(m)
        uniq[0] = fsort[0]                  # fsort[0] is unique
        ntot[0] = wsort[0]
        npos[0] += wsort[0] * ysort[0]
        num_uniq = 1
        for k in range(1, m):
            if fsort[k] != fsort[k - 1]:    # fsort[k] is new.
                uniq[num_uniq] = fsort[k]
                num_uniq += 1
            ntot[num_uniq - 1] += wsort[k]
            npos[num_uniq - 1] +=  wsort[k] * ysort[k]
        uniq = uniq[:num_uniq]
        npos = npos[:num_uniq]
        ntot = ntot[:num_uniq]
        
        # Get cumulative counts/positives/negatives for each possible split
        nleft = ntot.cumsum()
        nright = tot_wt - nleft
        npos_left = npos.cumsum()
        nneg_left = nleft - npos_left
        npos_right = ywt.sum() - npos_left
        nneg_right = nright - npos_right

        # trim to valid splits (at least min_leaf both sides)
        mask = (nleft >= min_leaf) & (nright >= min_leaf)
        nleft = nleft[mask]
        nright = nright[mask]
        npos_left = npos_left[mask]
        npos_right = npos_right[mask]
        nneg_left = nneg_left[mask]
        nneg_right = nneg_right[mask]

        # at this point there might be no valid splits
        if not mask.any():
            continue

        # Compute Gini impurity for each split
        gini_left = 1 - (npos_left/nleft)**2 - (nneg_left/nleft)**2
        gini_right = 1 - (npos_right/nright)**2 - (nneg_right/nright)**2
        gini_split = (nleft/tot_wt) * gini_left + (nright/tot_wt) * gini_right

        # Select the best split
        score = gini_split.min()
        if score < best_score:
            improve = True
            best_score = score
            best_feature = feature_idx
            # Need index of feature of left side of split in the uniq array
            # The gini_* arrays do not align with it, so subset with mask
            left_thr_idx = gini_split.argmin()
            left_thr = uniq[mask][left_thr_idx]
            split_idx = np.where(uniq==left_thr)[0][0]
            best_thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
    return (best_feature, best_thr)
