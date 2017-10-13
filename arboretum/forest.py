'''
Class Forest implements a random forest using the trees from tree.py.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .base import BaseModel


class Forest(BaseModel):
    '''
    Class Forest implements random forest classification and regression 
    models using trees from arboretum.tree.
    '''

    estimator_params = ['max_features', 'min_leaf', 'max_depth']

    def __init__(self, base_estimator, n_trees, max_features, min_leaf, max_depth):
        self.base_estimator = base_estimator
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_leaf = min_leaf
        self.max_depth = max_depth
    
    def _get_maxf(self):
        '''Get adjusted max_features value. Overridden in RFClassifier.'''
        return self.max_features

    def fit(self, x, y, weights=None):
        '''
        Fits a random forest using tree.Tree to the given data. 
        Also sets the oob_decision_function_ attribute.

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
        n = len(y)
        self.n_features_ = x.shape[1]
        self.estimators_ = []
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
        est_params['max_features'] = self._get_maxf()
        all_idx = np.arange(n)
        oob_ct = np.zeros(n)
        oob_prob = np.zeros(n)
        for k in range(self.n_trees):
            model = self.base_estimator.__class__(**est_params)
            boot_idx = np.random.randint(n, size=n)
            oob_idx = np.setdiff1d(all_idx, boot_idx)
            model.fit(x[boot_idx], y[boot_idx], weights=weights[boot_idx])
            self.estimators_.append(model)
            oob_ct[oob_idx] += 1
            oob_prob[oob_idx] += model.decision_function(x[oob_idx])
        # TODO: check for NaN
        self.oob_decision_function_ = oob_prob / oob_ct
        return self

    def decision_function(self, x):
        '''
        Returns the decision function for each row in x. In regression trees,
        this is an estimate. In classification trees, it is a probability.
        In either case, it is the average over the trees in this forest.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        self._predict_check(x)
        dv = np.zeros(len(x))
        for model in self.estimators_:
            dv += model.decision_function(x)
        return dv / self.n_trees


class RFRegressor(Forest):
    '''
    RFClassifier implements a random forest regression model using an 
    arboretum.tree.RegressionTree for its basis functions.
    This is a single-output model that minimizes mse.

    Args:
        n_trees: (int) number of trees to fit
        max_features: controls number of features to try at each split
            If float, should be in (0, 1]; use int(n_features * max_features)
            If int, use that number of features. If None, use all features.
        min_leaf: minimum number of samples for a leaf; default 5
        max_depth: (int) the maximum depth of the trees grown. 
            Default of None for no depth limit.
    '''
    def __init__(self, n_trees=30, max_features=None, min_leaf=5, max_depth=None):
        base_estimator = tree.RegressionTree()
        super().__init__(base_estimator, n_trees=n_trees,
                        max_features=max_features, min_leaf=min_leaf,
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


class RFClassifier(Forest):
    '''
    RFClassifier implements a random forest classifier using an 
    arboretum.tree.ClassificationTree for its basis functions.
    This is a single-output, binary classifier, using gini impurity.

    Args:
        n_trees: (int) number of trees to fit
        max_features: controls number of features to try at each split
            If float (0, 1]; use int(n_features * max_features) 
            If int, use that number of features. 
            If None, use np.round(np.sqrt(n_features)).
        min_leaf: minimum number of samples for a leaf; default 1
        max_depth: (int) the maximum depth of the trees grown.
            Default of None for no depth limit.
    '''

    def __init__(self, n_trees=30, max_features=None, min_leaf=1, max_depth=None):
        base_estimator = tree.ClassificationTree()
        super().__init__(base_estimator, n_trees=n_trees, 
                        max_features=max_features, min_leaf=min_leaf,
                        max_depth=max_depth)

    def _get_maxf(self):
        '''
        Get the adjusted value for max_features.
        NB: self.n_features_ has to be set before calling this.
        '''
        if self.max_features is None:
            return int(np.round(np.sqrt(self.n_features_)))
        return self.max_features

    def predict_proba(self, x):
        '''
        Predicts probabilities of the positive class for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        return self.decision_function(x)

    def predict(self, x):
        '''
        Predicts class membership for the rows in x. Predicted class is the one
        with the higest mean probability across the trees.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples, ) of class labels for each row
        '''
        return (self.predict_proba(x) > 0.5).astype(int)
