'''
Gradient Boosting models for least-squares and bernoulli models.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .base import BaseModel
from scipy.special import expit, logit


class GBM(BaseModel):
    '''
    GBM is a base class for gradient boosting models.

    Args:
        n_trees: (int) number of trees to fit
        learn_rate: the step size for the model
        max_depth: (int) the maximum depth of the trees grown.
        subsample: (float in (0.0, 1.0]) fraction of rows to sample 
            for training each tree
        max_features: (int) number of features to try at each split
    '''

    estimator_params = ['max_features', 'max_depth']

    def __init__(self, n_trees=100, learn_rate=0.1, max_depth=3,
                    subsample=1.0, max_features=-1):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def decision_function(self, x):
        '''
        Returns the decision function for each row in x. 
        In a regression model, this is the estimate of the targets. 
        In a classification model, it is the estimated log-odds of the
        positive class.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        self._predict_check(x)
        pred = np.zeros(len(x)) + self.f0
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return pred

    def staged_decision_function(self, x, predict_at=None):
        '''
        For each entry in predict_at, returns the decision function for each
        row of x, using only the first predict_at[k] trees of the model.
        Values in predict at should be ascending, and for all k:
        0 < predict_at[k] <= n_trees

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)
            predict_at: (list of int) a list of numbers of trees to use in 
                predicting with a partial model. Values should be in ascending 
                order and 0 < predict_at[k] <= n_trees for all k.

        Returns:
            array (n_samples, n_steps) decision function for each row in x
            at each iteration count in predict_at
        '''
        self._predict_check(x)
        if predict_at is None:
            predict_at = 1 + np.arange(len(self.estimators_))
        out = np.zeros((len(x), len(predict_at)))
        pred = np.zeros(len(x)) + self.f0
        j = 0
        for k, model in enumerate(self.estimators_, 1):
            pred += self.learn_rate * model.predict(x)
            if k == predict_at[j]:
                out[:, j] = pred
                j += 1
        return out


class GBRegressor(GBM):

    def fit(self, x, y, weights=None):
        '''
        Fits a least-squares gradient boosting model using x and y.

        Args:
            x: Training data features; ndarray of shape (n_samples, n_features)
            y: Training set labels; shape is (n_samples, )
            weights: sample weights; shape is (n_samples, )
                default is None for equal weights/unweighted

        Returns:
            Returns self, the fitted estimator
        '''
        n = len(y)
        self.n_features_ = x.shape[1]
        if weights is None:
            weights = np.ones_like(y)
        n_subsample = int(np.round(self.subsample * n))
        self.estimators_ = []
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
        self.f0 = np.average(y, weights=weights)
        r = y - self.f0
        for k in range(self.n_trees):
            model = tree.RegressionTree(**est_params)
            idx = np.random.choice(n, size=n_subsample, replace=False)
            model.fit(x[idx], r[idx], weights[idx])
            self.estimators_.append(model)
            step_k = self.learn_rate * model.predict(x)
            r = r - step_k
        return self

    def predict(self, x):
        '''
        Estimates target for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return self.decision_function(x)

    def staged_predict(self, x, predict_at=None):
        '''
        For each entry in predict_at, returns the regression estimate for each
        row of x, using only the first predict_at trees of the model.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)
            predict_at: (list of int) a list of numbers of trees to use in 
                predicting with a partial model. Values should be in ascending 
                order and 0 < predict_at <= n_trees.

        Returns:
            array (n_samples, n_steps) regression estimate for each row in x
            at each iteration count in predict_at
        '''
        return self.staged_decision_function(x, predict_at)


class GBClassifier(GBM):

    def fit(self, x, y, weights=None):
        '''
        Fits a binary classifier using gradient boosting and bernoulli loss.

        Args:
            x: Training data features; ndarray of shape (n_samples, n_features)
            y: Training set labels; shape is (n_samples, )
            weights: sample weights; shape is (n_samples, )
                default is None for equal weights/unweighted

        Returns:
            Returns self, the fitted estimator
        '''
        n = len(y)
        self.n_features_ = x.shape[1]
        if weights is None:
            weights = np.ones_like(y)
        self.estimators_ = []
        n_subsample = int(np.round(self.subsample * n))
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
        p = np.average(y, weights=weights)
        self.f0 = logit(p)               # We'll need this to predict
        r = y - p                        # initial residual
        f = self.f0                      # accumulated log-odds prediction
        for k in range(self.n_trees):
            model = tree.RegressionTree(**est_params)
            self.estimators_.append(model)
            idx = np.random.choice(n, size=n_subsample, replace=False)
            model.fit(x[idx], r[idx], weights[idx])
            # adjust leaf values (log-odds, in-subsample)
            leaves = model.apply(x[idx])
            num = np.bincount(leaves, weights=weights[idx] * r[idx])
            den_val = weights[idx] * (y[idx] - r[idx]) * (1 - y[idx] + r[idx])
            den = np.bincount(leaves, weights=den_val)
            den0idx = (np.abs(den) < 1e-100)
            den[den0idx] = 1.
            vals = np.where(den0idx, 0, num/den)
            model.value = vals
            # adjust current prediction (log-odds, all x):
            f += self.learn_rate * model.predict(x)
            # compute new residual y - expit(f):
            r = y - expit(f)
        return self

    def predict_proba(self, x):
        '''
        Predicts probabilities of the positve class for each row in x

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        return expit(self.decision_function(x))

    def predict(self, x):
        '''
        Estimates target label for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return (self.decision_function(x) > 0).astype(int)

    def staged_predict_proba(self, x, predict_at=None):
        '''
        For each entry in predict_at, returns the estimated probability for
        the positive class for each row of x, using only the first predict_at
        trees of the model.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)
            predict_at: (list of int) a list of numbers of trees to use in 
                predicting with a partial model. Values should be in ascending 
                order and 0 < predict_at <= n_trees.

        Returns:
            array (n_samples, n_steps) positive class probability for each row
            in x at each iteration count in predict_at
        '''
        return expit(self.staged_decision_function(x, predict_at))

