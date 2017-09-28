'''
Gradient Boosting models for least-squares and bernoulli models.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .basemodel import BaseModel
from scipy.special import expit, logit


class GBRegressor(BaseModel):

    estimator_params = ['max_features', 'max_depth']

    def __init__(self, n_trees=100, learn_rate=0.1,
                max_depth=3, subsample=1.0, max_features=-1):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def fit(self, x, y, weights=None):
        n = len(y)
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

    # NB: this fails if learn_rate is changed by field access between fit and predict
    def predict(self, x):
        pred = np.zeros(len(x)) + self.f0
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return pred

    def staged_predict(self, x, predict_at=None):
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


class GBClassifier(BaseModel):

    estimator_params = ['max_features', 'max_depth']

    def __init__(self, n_trees=100, learn_rate=0.1,
                max_depth=3, subsample=1.0, max_features=-1):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def fit(self, x, y, weights=None):
        n = len(y)
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

    def decision_function(self, x):
        pred = np.zeros(len(x)) + self.f0
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return pred

    def predict_proba(self, x):
        return expit(self.decision_function(x))

    def predict(self, x):
        return (self.decision_function(x) > 0).astype(int)

    def staged_decision_function(self, x, predict_at=None):
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

    def staged_predict_proba(self, x, predict_at=None):
        return expit(self.staged_decision_function(x, predict_at))

