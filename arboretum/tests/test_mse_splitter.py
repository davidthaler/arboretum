import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ..datasets import load_mtcars, load_als
from .. mse_splitter import split
from .. import tree_constants as tc


NO_SPLIT = (tc.NO_FEATURE, tc.NO_THR)
X, Y = load_mtcars()
W = np.ones_like(Y)
XTR, YTR, XTE, YTE = load_als()
WTR = np.ones_like(YTR)


class TestMseSplitter(unittest.TestCase):
    '''
    The mse_splitter is tested by comparing its output to the split made by
    a depth 1 sklearn.tree.DecisionTreeRegressor (a stump) on data from a few
    regression datasets.
    '''

    def stump(self, x, y, w=None, **kwargs):
        '''
        This method creates a fitted decision stump and returns the feature and
        threshold from it for use as ground truth.

        Args:
            x: training features, a 2-D numpy array
            y: training lables
            w: weights, default None for unweighted
            **kwargs: keyword argumentss for decision tree constructor
        
        Returns:
            2-tuple of the feature and threshold found by the decision stump
        '''
        dt = DecisionTreeRegressor(max_depth=1, **kwargs)
        dt.fit(x, y, sample_weight=w)
        tree = dt.tree_
        f = tree.feature[0]
        thr = tree.threshold[0]
        return (f, thr)

    # tests using the small mtcars regression dataset

    def test_mtcars(self):
        '''Check split found by mse_splitter matches sklearn on mtcars'''
        gold_f, gold_thr = self.stump(X, Y, W)
        f, thr = split(X, Y, W, X.shape[1], 1)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)

    def test_min_leaf(self):
        '''Check split found matches sklearn on mtcars when min_leaf=10'''
        gold_f, gold_thr = self.stump(X, Y, W, min_samples_leaf=10)
        f, thr = split(X, Y, W, X.shape[1], 10)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)

    def test_sample_weights(self):
        '''Check for same split when non-uniform weights used'''
        w = W.copy()
        w[::2] = 2
        gold_f, gold_thr = self.stump(X, Y, w)
        f, thr = split(X, Y, w, X.shape[1], 1)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)

    def test_pure_node(self):
        '''Test for no split and no error on a pure node'''
        y = np.ones_like(Y)
        result = split(X, y, W, X.shape[1], 1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_pure_data(self):
        '''Test for no split and no error on pure data'''
        result = split(np.ones_like(X), Y, W, X.shape[1], 1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_one_row(self):
        '''Test for no split and no error on a single row'''
        result = split(X[:1], Y[:1], W[:1], X.shape[1], 1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    # Tests on the larger ALS dataset

    def test_als(self):
        '''Check that split matches on larger data'''
        gold_f, gold_thr = self.stump(XTR, YTR, WTR)
        f, thr = split(XTR, YTR, WTR, XTR.shape[1], 1)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)

    def test_wtd_als(self):
        '''Check for same split with non-uniform weights'''
        w = WTR.copy()
        w[::2] = 2
        gold_f, gold_thr = self.stump(XTR, YTR, w)
        f, thr = split(XTR, YTR, w, XTR.shape[1], 1)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)

    def test_min_leaf_als(self):
        '''Check for same split with min_leaf set'''
        # NB: the slice :100 allows the min_leaf setting to change the split
        gold_f, gold_thr = self.stump(XTR[:100], YTR[:100], WTR[:100],
                                            min_samples_leaf=10)
        f, thr = split(XTR[:100], YTR[:100], WTR[:100], XTR.shape[1],
                                             min_leaf=10)
        self.assertEqual(gold_f, f)
        self.assertAlmostEqual(gold_thr, thr)
