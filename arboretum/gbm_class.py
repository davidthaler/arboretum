'''
First cut at a GBM classifier

author: David Thaler
date: September 2017
'''
import numpy as np
from . import tree
from .basemodel import BaseModel
from scipy.special import expit, logit
from . import tree_constants as tc

class GBClassifier(BaseModel):

    estimator_params = ['max_features', 'max_depth']

    def __init__(self, n_trees, learn_rate,  max_depth, subsample, max_features):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def fit(self, x, y):
        n = len(y)
        self.estimators_ = []
        n_subsample = int(np.round(self.subsample * n))
        est_params = {ep:getattr(self, ep) for ep in self.estimator_params}
        p = y.mean()            
        self.f0 = logit(p)               # We'll need this to predict
        r = y - p                        # initial residual
        f = self.f0                      # accumulated log-odds prediction
        for k in range(self.n_trees):
            model = tree.RegressionTree(**est_params)
            self.estimators_.append(model)
            idx = np.random.choice(n, size=n_subsample, replace=False)
            model.fit(x[idx], r[idx])
            # adjust leaf values (log-odds, in-subsample)
            leaves = model.apply(x[idx])
            num = np.bincount(leaves, weights=r[idx])
            den = np.bincount(leaves, 
                weights=(y[idx] - r[idx]) * (1 - y[idx] + r[idx]))
            den0idx = (np.abs(den) < 1e-100)
            den[den0idx] = 1.
            vals = np.where(den0idx, 0, num/den)
            model.tree_[:, tc.VAL_COL] = vals
            # adjust current prediction (log-odds, all x):
            f += self.learn_rate * model.predict(x)
            # compute new residual y - expit(f):
            r = y - expit(f)
        return self

    def predict_proba(self, x):
        pred = np.zeros(len(x)) + self.f0
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return expit(pred)
