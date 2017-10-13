'''
The split function here finds splits that minimize residual sum of squares.
It runs under numba for speed, since these are the innermost loops in 
decision tree fitting.

author: David Thaler
date: September 2017
'''
import numpy as np
import numba
from . import tree_constants as tc


@numba.jit(nopython=True)
def split(x, y, wts, max_features, min_leaf):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in residual sum of squares.
    Each side of the split must have at least min_leaf samples.

    Note:
        If no valid split is found after max_features, and max_features is less
        than the number of features, splitter will continue to try features, one
        at a time, in random order, until a valid split is found, or until all 
        features have been tried.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of regression targets
        wts: m-element 1-D array of sample weights, use ones if unweighted
        max_features: try up to this number of features per split
            Caller must set to value in 1...x.shape[1]
        min_leaf: minimum number of samples for a leaf node

    Returns:
        2-tuple of feature index and split threshold of best split.
    '''
    m, n = x.shape
    NO_SPLIT = (tc.NO_FEATURE, tc.NO_THR)
    improve = False
    tot_wt = wts.sum()
    yw = wts * y
    tot_yw = yw.sum()
    yyw = y * yw
    tot_yyw = yyw.sum()
    mu = tot_yw / tot_wt
    node_score = (wts * ((y - mu)**2)).sum()
    if node_score==0:
        return NO_SPLIT
    # filling with a 'max-value'
    results = node_score * np.ones((n, 2))
    col_order = np.random.choice(np.arange(n), size=n, replace=False)
    for col_ct in range(n):
        if col_ct >= max_features and improve:
            break
        feature_idx = col_order[col_ct]
        f = x[:, feature_idx]

        # Produce 4 arrays:
        # 1) sorted unique values in f
        # 2) count of each unique value (usually 1)
        # 3) sum of targets for each unique
        # 4) sum of squared targets for each unique
        uniq = np.zeros(m)
        ntot = np.zeros(m)
        ysum = np.zeros(m)
        yssq = np.zeros(m)
        cur_val = np.nan
        num_uniq = 0
        a = -1
        b = -1
        f_idx = np.argsort(f)
        for i in range(m):
            idx = f_idx[i]
            if f[idx] != cur_val:
                cur_val = f[idx]
                uniq[num_uniq] = cur_val
                num_uniq += 1
            # the count on the left, i + 1,  matches min_leaf
            if (i + 1) == min_leaf:
                a = num_uniq - 1
            # m - (i+1) just dropped below min_leaf
            if (m - i) == min_leaf:
                b = num_uniq - 1
            ntot[num_uniq - 1] += wts[idx]
            ysum[num_uniq - 1] += yw[idx]
            yssq[num_uniq - 1] += yyw[idx]
        uniq = uniq[:num_uniq]
        ntot = ntot[:num_uniq]
        ysum = ysum[:num_uniq]
        yssq = yssq[:num_uniq]

        # at this point there might be no valid splits
        if b <= a:
            continue

        # Get cumulative counts/sum/ssq for each possible split
        nleft = ntot.cumsum()
        nright = tot_wt - nleft
        ysum_left = ysum.cumsum()
        ysum_right = tot_yw - ysum_left
        yssq_left = yssq.cumsum()
        yssq_right = tot_yyw - yssq_left

        nleft = nleft[a:b]
        nright = nright[a:b]
        ysum_left = ysum_left[a:b]
        ysum_right = ysum_right[a:b]
        yssq_left = yssq_left[a:b]
        yssq_right = yssq_right[a:b]

        # Compute combined mse for each split
        sk_left = yssq_left - (ysum_left**2) / nleft
        sk_right = yssq_right - (ysum_right**2) / nright
        sk = sk_left + sk_right

        # Select the best split
        split_pos = sk.argmin()
        split_score = sk[split_pos]
        split_idx = a + split_pos
        thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
        results[feature_idx] = (split_score, thr)
        if split_score < node_score:
            improve = True
    best_split_idx = results[:, 0].argmin()
    best_score = results[best_split_idx, 0]
    if best_score < node_score:
        best_thr = results[best_split_idx, 1]
        return (best_split_idx, best_thr)
    else:
        return NO_SPLIT
