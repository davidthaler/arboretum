import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ..datasets import load_iris
from ..gini_splitter import split
from .. import tree_constants as tc

# Hand-built toy data

A = np.array([[2., 2., 3., 3., 2.], 
             [2., 3., 3., 4., 2.], 
             [2., 2., 3., 4., 3.],
             [2., 3., 4., 4., 3.]])

B = np.array([0., 0., 1., 1.])
W = np.ones_like(B)
NO_SPLIT = (tc.NO_FEATURE, tc.NO_THR)
PLACES = 4
IRIS = load_iris()

class TestGiniSplitter(unittest.TestCase):
    '''
    First test the gini_splitter on a tiny hand-checked toy problem.
    Then test it on real data, using a depth 1 sklearn DecisionTreeClassifier
    for a ground truth. In doing so, we have to force unique splits, because 
    otherwise the result is non-deterministic.
    '''

    # Basic tests on the hand-built data

    def test_one_row(self):
        '''Test for no split and no error when data has only one row'''
        result = split(A[:1], B[:1], W[:1],
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_pure_node(self):
        '''Test for no split and no error if node is pure'''
        result = split(A[:2], B[:2], W[:1],
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_pure_feature(self):
        '''Test for no split and no error if x is pure'''
        result = split(A[:, :1], B, W,
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_no_improve(self):
        '''Test for no split when only a no-gain split exists'''
        result = split(A[:,:2], B, W,
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertTupleEqual(result, NO_SPLIT)

    def test_wtd_improve(self):
        '''Test that split from no_improve can exist if weighted'''
        w = np.array([1,1,1,2])
        f, thr = split(A[:,:2], B, w,
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertEqual(f, 1)
        self.assertAlmostEqual(thr, 2.5, places=PLACES)

    def test_perfect(self):
        '''Test for perfect split in column 4 with threshold of 2.5'''
        f, thr = split(A, B, W,
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertEqual(f, 4)
        self.assertAlmostEqual(thr, 2.5, places=PLACES)

    def test_imperfect(self):
        '''Test for the imperfect split in column 2'''
        f, thr = split(A[:, :3], B, W,
                    max_features=A.shape[1], min_leaf=1)[:2]
        self.assertEqual(f, 2)
        self.assertAlmostEqual(thr, 3.5, places=PLACES)

    # Checking splitter against sklearn on iris data 

    def stump(self, x, y, w=None, **kwargs):
        '''This method gives us the ground truth for a split'''
        dt = DecisionTreeClassifier(max_depth=1, **kwargs)
        dt.fit(x, y, sample_weight=w)
        f = dt.tree_.feature[0]
        thr = dt.tree_.threshold[0]
        return (f, thr)

    # NB: Splitting iris data on columns 2 or 3 gives the same partition,
    # which makes the test non-deterministic, so we use one at a time.

    def test_success(self):
        '''Check the main success scenario on iris data.'''
        x, y = IRIS
        w = np.ones_like(y)
        f, thr = split(x[:, :-1], y, w, max_features=3,
                        min_leaf=1)[:2]
        gold = self.stump(x[:, :-1], y, w)
        self.assertEqual(f, gold[0])
        self.assertAlmostEqual(thr, gold[1])
        f, thr = split(x[:, [0, 1, 3]], y, w, max_features=3, min_leaf=1)[:2]
        gold = self.stump(x[:, [0, 1, 3]], y, w)
        self.assertEqual(f, gold[0])
        self.assertAlmostEqual(thr, gold[1], places=PLACES)

    # AFAIK: the splits below are deterministic, given the parameters

    def test_weights(self):
        '''Check for same split when weights are used.'''
        x, y = IRIS
        w = np.ones_like(y)
        w[75:] += 1
        result = split(x, y, w, max_features=x.shape[1], min_leaf=1)[:2]
        gold = self.stump(x, y, w)
        self.assertEqual(result[0], gold[0])
        self.assertAlmostEqual(result[1], gold[1], places=PLACES)

    def test_min_leaf(self):
        '''Check that the same split is found with min_leaf set'''
        x, y = IRIS
        x, y = x[::10], y[::10]
        w = np.ones_like(y)
        result = split(x, y, w, max_features = x.shape[1], min_leaf=4)[:2]
        gold = self.stump(x, y, w, min_samples_leaf=4)
        self.assertEqual(result[0], gold[0])
        self.assertAlmostEqual(result[1], gold[1], places=PLACES)
