'''
Object wrapper for sklearn compatibility.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree_builder
from . import mse_splitter
from . import gini_splitter
from .base import BaseModel
from . import tree_constants as tc


class Tree(BaseModel):
    '''
    Tree is a superclass for trees. It takes a splitter as a constructor
    parameter that determines what criterion it minimizes.

    Args:
        split_fn: function or callable that takes training data, labels, 
            sample weights and the min_leaf and max_features parameters
            (explained below) and returns the best split feature and threshold.
        max_features: controls number of features to try at each split
            If float, should be in (0, 1]; use int(n_features * max_features)
            If int, use that number of features. If None, use all features.
        min_leaf: if weights are passed to fit(), this is the minimum sample
            weight in a leaf node; if unweighted, it is the minimum number 
            of samples in a leaf node.
        max_depth: (int) the maximum depth of this tree. 
            Default of None for no depth limit.
    '''
    def __init__(self, split_fn, max_features=None, min_leaf=1, max_depth=None):
        self.split_fn = split_fn
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.max_depth = max_depth

    def _get_maxf(self):
        '''
        Get the correct int value for max_features
        NB: self.n_features_ has to be set before calling this.
        '''
        if type(self.max_features) is float:
            return int(self.n_features_ * self.max_features)
        elif type(self.max_features) is int:
            return self.max_features
        elif self.max_features is None:
            return self.n_features_
        else:
            msg = '%s not valid for self.max_features' % self.max_features
            raise AttributeError(msg)

    def fit(self, x, y, weights=None):
        '''
        Fits classification or regression trees using x, y and the weights.

        Args:
            x: Training data features; ndarray of shape (n_samples, n_features)
            y: Training set labels; shape is (n_samples, )
            weights: sample weights; shape is (n_samples, )
                default is None for equal weights/unweighted

        Returns:
            Returns self, the fitted estimator
        '''
        if weights is None:
            weights = np.ones_like(y)
        max_depth = -1 if self.max_depth is None else self.max_depth
        self.n_features_ = x.shape[1]
        self.tree_ = tree_builder.build_tree(x, y, self.split_fn,
                                             wts=weights,
                                             max_features=self._get_maxf(),
                                             min_leaf=self.min_leaf,
                                             max_depth=max_depth)
        return self

    def apply(self, x):
        '''
        Finds the node number in the tree that each instance in x lands in.

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype int) of leaf node numbers for each point in x.
        '''
        self._predict_check(x)
        return tree_builder.apply(self.tree_, x)

    def decision_function(self, x):
        '''
        Returns the decision function for each row in x. 
        In a regression model, this is the estimate of the targets. 
        In a classification model, it is the estimated probability of the
        positive class.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        self._predict_check(x)
        return tree_builder.prediction_value(self.tree_, x)

    @property
    def value(self):
        '''
        The decision function value for each leaf node in this tree.

        Returns:
            1-D numpy array of size (n_nodes) of leaf decision values
        '''
        return self.tree_[:, tc.VAL_COL]

    @value.setter
    def value(self, new_val):
        '''
        Set the decision function value for each leaf node in this tree.

        Args:
            new_val: 1-D numpy array of size (n_nodes, ) of new leaf values
        '''
        if type(new_val) is not np.ndarray:
            raise ValueError('new value must be ndarray')
        if len(new_val) != len(self.tree_):
            raise ValueError('new value has wrong shape')
        self.tree_[:, tc.VAL_COL] = new_val


class RegressionTree(Tree):
    '''
    RegressionTree implements a regression tree minimizing mean squared error.

    Args:
        max_features: (int) number of features to try at each split
        min_leaf: if weights are passed to fit(), this is the minimum sample
            weight in a leaf node; if unweighted, it is the minimum number 
            of samples in a leaf node.
        max_depth: (int) the maximum depth of this tree.
            Default of None for no depth limit.
    '''

    def __init__(self, max_features=None, min_leaf=1, max_depth=None):
        super().__init__(split_fn=mse_splitter.split, 
                         max_features=max_features, 
                         min_leaf=min_leaf,
                         max_depth=max_depth)

    def predict(self, x):
        '''
        Estimates target for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return self.decision_function(x)


class ClassificationTree(Tree):
    '''
    ClassificationTree is a classification tree minimizing gini impurity.

    Args:
        max_features: (int) number of features to try at each split
        min_leaf: if weights are passed to fit(), this is the minimum sample
            weight in a leaf node; if unweighted, it is the minimum number 
            of samples in a leaf node.
        max_depth: (int) the maximum depth of this tree.
            Default of None for no depth limit.
    '''

    def __init__(self, max_features=None, min_leaf=1, max_depth=None):
        super().__init__(split_fn=gini_splitter.split, 
                         max_features=max_features, 
                         min_leaf=min_leaf,
                         max_depth=max_depth)

    def predict_proba(self, x):
        '''
        Predicts probabilities of the positve class for each row in x

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        return self.decision_function(x)

    def predict(self, x):
        '''
        Predicts class membership for the rows in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples, ) of class labels for each row
        '''
        return (self.predict_proba(x) > 0.5).astype(int)
