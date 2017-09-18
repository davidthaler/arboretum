'''
Object wrapper for sklearn compatibility.

author: David Thaler
date: September 2017
'''
import numpy as np
import tree_builder
import mse_splitter
import gini_splitter


class Tree:

    def __init__(self, split_fn, max_features=None, 
                min_leaf=1, min_split=2, max_depth=None):
        self.split_fn = split_fn
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
            raise ValueError('Tree fitted for %d-dimensions, but data has %d' 
                                % (self.n_features, x.shape[1]))

    def fit(self, x, y):
        self.n_features_ = x.shape[1]
        self.tree_ = tree_builder.build_tree(x, y, self.split_fn,
                                             max_features=self.max_features,
                                             min_leaf=self.min_leaf,
                                             min_split=self.min_split,
                                             max_depth=self.max_depth)
        return self

    def apply(self, x):
        self._check_x(x)
        return tree_builder.apply(self.tree_, x)

    def prediction_value(self, x):
        '''For classifiers, a prob. For regressors, an estimate.'''
        self._check_x(x)
        return tree_builder.prediction_value(self.tree_, x)


class RegressionTree(Tree):

    def __init__(self, max_features=None, min_leaf=1, min_split=2, max_depth=None):
        super().__init__(split_fn=mse_splitter.split, 
                         max_features=max_features, 
                         min_leaf=min_leaf, 
                         min_split=min_split, 
                         max_depth=max_depth)

    def predict(self, x):
        return self.prediction_value(x)


class ClassificationTree(Tree):

    def __init__(self, max_features=None, min_leaf=1, min_split=2, max_depth=None):
        super().__init__(split_fn=gini_splitter.split, 
                         max_features=max_features, 
                         min_leaf=min_leaf, 
                         min_split=min_split, 
                         max_depth=max_depth)

    def predict_proba(self, x):
        return self.prediction_value(x)

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype(int)
