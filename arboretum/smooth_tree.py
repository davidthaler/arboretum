'''
Smoothed decision tree models

author: David Thaler
date: October 2017
'''
import numpy as np
from . import mse_splitter
from . import gini_splitter
from . import tree_constants as tc
from . import Tree

class SmoothTree(Tree):
    '''
    SmoothTree is a superclass for smoothed decision trees. 
    If the raw response for node n is y_n, then the smoothed response is:

        y_n if n=0 (root node)
        (w_n * y_n + y_parent * vss) / (w_n + vss) for n != 0

    where w_n is the weight (or sample count) of node n, vss is the virtual 
    sample size parameter and y_parent is the response of the parent of node n.

    In this class, fit() and apply are inherited. Smoothing is 'lazy' - 
    the predictions are only smoothed when a prediction function is called.

    Args:
        vss: a virtual sample size; the parent response is averaged together 
            with each node value weighted by vss and this node's count/weight
        split_fn: function or callable that takes training data, labels, 
            sample weights and the min_leaf and max_features parameters
            (explained below) and returns the best split feature and threshold.
        max_features: controls number of features to try at each split
            If float, should be in (0, 1]; use int(n_features * max_features)
            If int, use that number of features. If None, use all features.
        min_leaf: minimum number of samples for a leaf; default 1
        max_depth: (int) the maximum depth of this tree. 
            Default of None for no depth limit.
    '''
    def __init__(self, vss, split_fn, max_features=None, min_leaf=1, max_depth=None):
        self.vss = vss
        super().__init__(split_fn=split_fn,
                         max_features=max_features,
                         min_leaf=min_leaf,
                         max_depth=max_depth)
    
    def decision_function(self, x):
        '''
        Returns the smoothed decision function for each row in x. 
        In a regression model, this is the estimate of the targets. 
        In a classification model, it is the estimated probability of the
        positive class.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        leaf_idx = self.apply(x)
        return self.value[leaf_idx]
    
    @property
    def value(self):
        '''
        Get the smoothed values for each node in the tree, both leaves 
        and internal nodes.

        Returns:
            array (number of nodes, ) of smoothed node values.
        '''
        vals = self.tree_[:, tc.VAL_COL]
        wts = self.tree_[:, tc.CT_COL]
        out = np.zeros_like(vals)
        out[0] = vals[0]
        for i in range(len(vals)):
            l = int(self.tree_[i, tc.CHILD_LEFT_COL])
            r = int(self.tree_[i, tc.CHILD_RIGHT_COL])
            if l != tc.NO_CHILD:
                out[l] = ((self.vss * out[i] + wts[l] * vals[l]) 
                            / (self.vss + wts[l]))
                out[r] = ((self.vss * out[i] + wts[r] * vals[r]) 
                            / (self.vss + wts[r]))
        return out


class SmoothRegressionTree(SmoothTree):
    '''
    Class SmoothRegressionTree fits a smoothed regression tree, in which the
    node values are smoothed back to the parent node values (which are also
    smoothed). See the doc for the SmoothTree class for the formula.

    Args:
        vss: a virtual sample size; the parent response is averaged together 
            with each node value weighted by vss and this node's count/weight
            Default of 0 is a regular tree.
        max_features: controls number of features to try at each split
            If float, should be in (0, 1]; use int(n_features * max_features)
            If int, use that number of features. If None, use all features.
        min_leaf: minimum number of samples for a leaf; default 5
        max_depth: (int) the maximum depth of this tree. 
            Default of None for no depth limit.
    '''
    def __init__(self, vss=0, max_features=None, min_leaf=5, max_depth=None):
        super().__init__(vss=vss,
                         split_fn=mse_splitter.split,
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


class SmoothClassificationTree(SmoothTree):
    '''
    Class SmoothClassificationTree fits a smoothed classification tree, one
    in which the node values are probabilities, and these are smoothed back
    towards the parent node value (which is also smoothed). See the doc for
    the SmoothTree class for the formula.

    Args:
        vss: a virtual sample size; the parent response is averaged together 
            with each node value weighted by vss and this node's count/weight
            Default of 0 is a regular tree.
        max_features: controls number of features to try at each split
            If float, should be in (0, 1]; use int(n_features * max_features)
            If int, use that number of features. If None, use all features.
        min_leaf: minimum number of samples for a leaf; default 1
        max_depth: (int) the maximum depth of this tree. 
            Default of None for no depth limit.
    '''
    def __init__(self, vss=0, max_features=None, min_leaf=1, max_depth=None):
        super().__init__(vss=vss,
                         split_fn=gini_splitter.split,
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
