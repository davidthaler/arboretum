'''
First cut at a least-squares GBM.

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .basemodel import BaseModel


class GBM(BaseModel):

    estimator_params = ['max_features', 'max_depth']

    def __init__(self, n_trees, learn_rate,  max_depth, subsample, max_features):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def fit(self, x, y):
        n = len(y)
        n_subsample = int(np.round(self.subsample * n))
        self.estimators_ = []
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
        r = y
        for k in range(self.n_trees):
            model = tree.RegressionTree(**est_params)
            idx = np.random.choice(n, size=n_subsample, replace=False)
            model.fit(x[idx], r[idx])
            self.estimators_.append(model)
            step_k = self.learn_rate * model.predict(x)
            r = r - step_k
        return self

    # NB: this fails if learn_rate is changed by field access between fit and predict
    def predict(self, x):
        pred = np.zeros(len(x))
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return pred
