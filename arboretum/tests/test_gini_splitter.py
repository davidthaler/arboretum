import unittest
import numpy as np
import numpy.testing as nt
from sklearn.tree import DecisionTreeClassifier
from ..datasets import load_iris
from ..gini_splitter import split
from .. import tree_constants as tc


class TestGiniSplitter(unittest.TestCase):
    '''
    First we test the gini_splitter on a tiny hand-checked toy problem.
    Then we try it on real data, using a depth 1 sklearn DecisionTreeClassifier
    for a ground truth. In doing so, we have to force unique splits, because 
    otherwise the result is non-deterministic.
    '''

    A = np.array([[2., 2., 3., 3., 2.], 
                 [2., 3., 3., 4., 2.], 
                 [2., 2., 3., 4., 3.],
                 [2., 3., 4., 4., 3.]])
    B = np.array([0., 0., 1., 1.])
    W = np.ones_like(B)
    NO_SPLIT = (tc.NO_FEATURE, tc.NO_THR)


    # Basic tests on the hand-built data

    def test_one_row(self):
        '''Test for no split and no error when data has only one row'''
        f, thr = split(self.A[:1], self.B[:1], self.W[:1])[:2]
        self.assertEqual(f, tc.NO_FEATURE)
        self.assertEqual(thr, tc.NO_THR)

    def test_pure_node(self):
        '''Test for no split and no error if node is pure'''
        f, thr = split(self.A[:2], self.B[:2], self.W[:1])[:2]
        self.assertTupleEqual((f, thr), (tc.NO_FEATURE, tc.NO_THR))

    def test_pure_feature(self):
        '''Test for no split and no error if x is pure'''
        f, thr = split(self.A[:, :1], self.B, self.W)[:2]
        self.assertTupleEqual((f, thr), ((tc.NO_FEATURE, tc.NO_THR)))

    def test_no_improve(self):
        '''
        Test for no split and no error when a split exists, 
        but no improvement is possible
        '''
        result = split(self.A[:, :1], self.B, self.W)[:2]
        self.assertTupleEqual(result, self.NO_SPLIT)

    def test_perfect(self):
        '''Test for perfect split in column 4 with threshold of 2.5'''
        f, thr = split(self.A, self.B, self.W)[:2]
        self.assertEqual(f, 4)
        self.assertAlmostEqual(thr, 2.5)

    def test_imperfect(self):
        '''Test for the imperfect split in column 2'''
        f, thr = split(self.A[:, :3], self.B, self.W)
        self.assertEqual(f, 2)
        self.assertAlmostEqual(thr, 3.5)

    # Checking splitter against sklearn on iris data 
    # NB: Splitting iris data on columns 2 or 3 gives the same partition,
    # which makes the test non-deterministic, so we use one at a time.

    def stump(self, x, y, w=None):
        '''This method gives us the ground truth for a split'''
        if w is None:
            w = np.ones_like(y)
        dt = DecisionTreeClassifier(max_depth=1)
        dt.fit(x, y, sample_weight=w)
        f = dt.tree_.feature[0]
        thr = dt.tree_.threshold[0]
        return (f, thr)

    def test_success(self):
        '''Check the main success scenario on iris data.'''
        x, y = load_iris()
        w = np.ones_like(y)
        f, thr = split(x[:, :-1], y, w)[:2]
        gold = self.stump(x[:, :-1], y, w)
        self.assertEqual(f, gold[0])
        self.assertAlmostEqual(thr, gold[1])
        f, thr = split(x[:, [0, 1, 3]], y, w)[:2]
        gold = self.stump(x[:, [0, 1, 3]], y, w)
        self.assertEqual(f, gold[0])
        self.assertAlmostEqual(thr, gold[1])

    # AFAIK: the split is deterministic, given these weights
    def test_weights(self):
        '''Check for same split when weights are used.'''
        x, y = load_iris()
        w = np.ones_like(y)
        w[75:] += 1
        result = split(x, y, w)
        gold = self.stump(x, y, w)
        self.assertEqual(result[0], gold[0])
        self.assertAlmostEqual(result[1], gold[1])
