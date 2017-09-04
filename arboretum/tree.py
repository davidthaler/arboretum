'''
Object wrapper for sklearn compatibility.

author: David Thaler
date: September 2017
'''
import numpy as np
import tree_builder

class Tree:

    def __init__(self, max_features=None, min_leaf=1, min_split=2, max_depth=None):
        self.min_leaf = min_leaf
        self.min_split = min_split
        self.max_features = -1 if max_features is None else max_features
        self.max_depth = -1 if max_depth is None else max_depth

    def _check_x(self, x):
        if type(x) is not np.ndarray:
            raise ValueError('Tree requires numpy array input')
        if x.ndim != 2:
            raise ValueError('Prediction input must be 2-D array')
        if not hasattr(self, 'n_features_'):
            raise Exception('Predict called before fit.')
        if x.shape[1] != self.n_features_:
            raise ValueError('Tree fitted for %d-dimensions, but data has %d')

    def fit(self, x, y):
        self.n_features_ = x.shape[1]
        self.tree_ = tree_builder.build_tree(x, y, 
                                            max_features=self.max_features,
                                            min_leaf=self.min_leaf,
                                            min_split=self.min_split,
                                            max_depth=self.max_depth)
        return self

    def apply(self, x):
        self._check_x(x)
        return tree_builder.apply(self.tree_, x)

    def predict(self, x):
        self._check_x(x)
        return tree_builder.predict(self.tree_, x)

    def predict_proba(self, x):
        self._check_x(x)
        return tree_builder.predict_proba(self.tree_, x)
