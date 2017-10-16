'''
Load functions for datasets that load and split data, ensuring correct
dtype for arboretum (ndarray of float), and splitting larger datasets
into train/test folds.

author: David Thaler
date: September 2017
'''
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')


def load_iris(target=1):
    '''
    Loads the irises data as a binary classification problem 
    with one of the classes as the target. 
    This is a small problem suitable for smoke testing.

    Args:
        target: class number (0, 1 or 2) of the positive class

    Returns:
        data, x, of shape (150 x 4) and labels y in {0,1} shape (150,)
    '''
    path = os.path.join(DATA_DIR, 'iris.csv')
    iris = pd.read_csv(path)
    y = iris.y.values
    y = (y == target).astype(float)
    x = iris.values[:, 1:]
    return x, y


def load_mtcars():
    '''
    Loads the mtcars data, a small regression problem.

    Returns:
        data, x, of shape (32 x 10) and targets, y, of shape (32,)
    '''
    path = os.path.join(DATA_DIR, 'mtcars.csv')
    mtcars = pd.read_csv(path)
    y = mtcars.mpg.values
    x = mtcars.drop(['CarName', 'mpg'], axis=1).values
    return x, y


def load_spam():
    '''
    Loads the spam data, a medium-sized binary classification problem.
    Data is split into training (n=3065) and test (n=1536) folds.

    Returns:
        xtrain (3065x57), ytrain (3065,), xtest (1565x57), ytest(1565,)
    '''
    path = os.path.join(DATA_DIR, 'spam.csv.gz')
    spam = pd.read_csv(path)
    xtr = spam[~spam.testid].values.astype(float)[:, 2:]
    xte = spam[spam.testid].values.astype(float)[:, 2:]
    ytr = spam.spam[~spam.testid].values.astype(float)
    yte = spam.spam[spam.testid].values.astype(float)
    return xtr, ytr, xte, yte


def load_als():
    '''
    Loads the als (Lou Gehrig's) data, a medium-sized regression problem.
    Data is split into training (n=1197) and test (n=625) folds.

    Returns:
        xtrain (1197x369), ytrain (1197,), xtest (625x369), ytest(625,)
    '''
    path = os.path.join(DATA_DIR, 'als.csv.gz')
    als = pd.read_csv(path)
    xtr = als[~als.testset].values.astype(float)[:, 2:]
    xte = als[als.testset].values.astype(float)[:, 2:]
    ytr = als.dFRS[~als.testset].values.astype(float)
    yte = als.dFRS[als.testset].values.astype(float)
    return xtr, ytr, xte, yte

def load_diabetes():
    '''
    Loads the diabetes dataset and splits it into train and test sets,
    with every 3rd row in test.

    Returns:
        xtrain, ytrain, xtest, ytest
    '''
    path = os.path.join(DATA_DIR, 'diabetes.csv.gz')
    data = pd.read_csv(path)
    idx = np.tile([True, True, False], 150)[:len(data)]
    xtr = data.values[idx, :-1]
    ytr = data.prog.values[idx]
    xte = data.values[~idx, :-1]
    yte = data.prog.values[~idx]
    return xtr, ytr, xte, yte
