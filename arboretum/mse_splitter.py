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


def split(x, y, max_features=-1, min_leaf=1):
    m, n = x.shape
    best_feature = tc.NO_FEATURE
    best_thr = tc.NO_THR
    improve = False
    if max_features < 1:
        max_features = n
    ym = y.mean()
    best_score = ((y - ym)**2).sum()
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
        sort_idx = f.argsort()
        fsort = f[sort_idx]
        ysort = y[sort_idx]
        ntot = np.zeros(m)
        uniq = np.zeros(m)
        ysum = np.zeros(m)
        yssq = np.zeros(m)
        uniq[0] = fsort[0]                  # fsort[0] is unique
        num_uniq = 1
        ntot[0] = 1
        ysum[0] += ysort[0]
        yssq[0] += ysort[0]**2
        for k in range(1, m):
            if fsort[k] != fsort[k-1]:      # fsort[k] is new
                uniq[num_uniq] = fsort[k]
                num_uniq += 1
            ntot[num_uniq - 1] += 1
            ysum[num_uniq - 1] += ysort[k]
            yssq[num_uniq - 1] += ysort[k]**2
        uniq = uniq[:num_uniq]
        ysum = ysum[:num_uniq]
        yssq = yssq[:num_uniq]
        ntot = ntot[:num_uniq]
        
        # Get cumulative counts/sum/ssq for each possible split
        nleft = ntot.cumsum()
        nright = m - nleft
        ysum_left = ysum.cumsum()
        ysum_right = y.sum() - ysum_left
        yssq_left = yssq.cumsum()
        yssq_right = yssq.sum() - yssq_left

        # trim to valid splits (at least min_leaf both sides)
        mask = (nleft >= min_leaf) & (nright >= min_leaf)
        nleft = nleft[mask]
        nright = nright[mask]
        ysum_left = ysum_left[mask]
        ysum_right = ysum_right[mask]
        yssq_left = yssq_left[mask]
        yssq_right = yssq_right[mask]

        # at this point there might be no valid splits
        if not mask.any():
            continue

        # Compute combined mse for each split
        sk_left = yssq_left - (ysum_left**2) / nleft
        sk_right = yssq_right - (ysum_right**2) / nright
        sk = sk_left + sk_right

        # Select the best split
        score = sk.min()
        if score < best_score:
            improve = True
            best_score = score
            best_feature = feature_idx
            # Need index of feature of left side of split in the uniq array
            # The sk_* arrays do not align with it, so subset with mask
            left_thr_idx = sk.argmin()
            left_thr = uniq[mask][left_thr_idx]
            split_idx = np.where(uniq==left_thr)[0][0]
            best_thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
    return (best_feature, best_thr)
