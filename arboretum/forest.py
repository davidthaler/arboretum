'''
Class Forest implements a random forest using the trees from tree.py.

author: David Thaler
date: September 2017
'''
import numpy as np
from tree import Tree


class Forest:

    def __init__(self, n_trees=30, max_features=None, 
                min_leaf=1, min_split=2, max_depth=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_leaf = min_leaf
        self.min_split = min_split
        self.max_depth = max_depth
        self.estimator_params = ('max_features', 'min_leaf', 
                                'min_split', 'max_depth')
    
    def fit(self, x, y):
        # check input
        n = len(y)
        self.estimators_ = []
        params = {ep:getattr(self, ep) for ep in self.estimator_params}
        all_idx = np.arange(n)
        oob_ct = np.zeros(n)
        oob_prob = np.zeros(n)
        for k in range(self.n_trees):
            model = Tree(**params)
            boot_idx = np.random.randint(n, size=n)
            oob_idx = np.setdiff1d(all_idx, boot_idx)
            model.fit(x[boot_idx], y[boot_idx])
            self.estimators_.append(model)
            oob_ct[oob_idx] += 1
            oob_prob[oob_idx] += model.predict_proba(x[oob_idx])
        # TODO: check for NaN
        self.oob_decision_function_ = oob_prob / oob_ct
        return self

    def predict_proba(self, x):
        # check input
        prob = np.zeros(len(x))
        for model in self.estimators_:
            prob += model.predict_proba(x)
        return prob / self.n_trees

    def predict(self, x):
        # check input
        return self.predict_proba(x).argmax(axis=1)
