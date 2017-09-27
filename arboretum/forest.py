'''
Class Forest implements a random forest using the trees from tree.py.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .basemodel import BaseModel


class Forest(BaseModel):
    '''
    Class Forest implements a random forest classifier using tree.Tree.
    This is a single-output, binary classifier, using gini impurity.
    '''

    estimator_params = ['max_features', 'min_leaf', 'min_split', 'max_depth']

    def __init__(self, base_estimator, n_trees, max_features, min_leaf, 
                    min_split, max_depth):
        self.base_estimator = base_estimator
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_leaf = min_leaf
        self.min_split = min_split
        self.max_depth = max_depth
    
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
            Returns self.
        '''
        if weights is None:
            weights = np.ones_like(y)
        # check input
        n = len(y)
        self.estimators_ = []
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
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
            oob_prob[oob_idx] += model.prediction_value(x[oob_idx])
        # TODO: check for NaN
        self.oob_decision_function_ = oob_prob / oob_ct
        return self


class RFRegressor(Forest):

    def __init__(self, n_trees=30, max_features=None, min_leaf=1, 
                    min_split=2, max_depth=None):
        base_estimator = tree.RegressionTree()
        super().__init__(base_estimator, n_trees=n_trees,
                        max_features=max_features, min_leaf=min_leaf,
                        min_split=min_split, max_depth=max_depth)

    def predict(self, x):
        '''
        Estimates target for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        pred = np.zeros(len(x))
        for model in self.estimators_:
            pred += model.predict(x)
        return pred / self.n_trees

class RFClassifier(Forest):

    def __init__(self, n_trees=30, max_features=None, min_leaf=1, 
                    min_split=2, max_depth=None):
        base_estimator = tree.ClassificationTree()
        super().__init__(base_estimator, n_trees=n_trees, 
                        max_features=max_features, min_leaf=min_leaf,
                        min_split=min_split, max_depth=max_depth)

    def predict_proba(self, x):
        '''
        Predicts probabilities of the two classes as the mean probability 
        predicted by each tree in the forest.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        # check input
        prob = np.zeros(len(x))
        for model in self.estimators_:
            prob += model.predict_proba(x)
        return prob / self.n_trees


    def predict(self, x):
        '''
        Predicts class membership for the rows in x. Predicted class is the one
        with the higest mean probability across the trees.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples, ) of class labels for each row
        '''
        # check input
        return (self.predict_proba(x) > 0.5).astype(int)
