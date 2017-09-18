# Arboretum
A collection of decision trees and tree ensembles (RF and GBM) in Python, accelerated with Numba.

## Goals
The initial goal of this project is to produce an all python library of 
decision trees and the most common tree ensemble models - random forest 
and GBM - that are:

* understandable by average programmers
* hackable by average programmers
* as accurate as the corresponding scikit-learn classes
* fast enough to actually be used (within 2x of scikit-learn)

Later we will use this as a test bed for tree and tree ensemble experiments.
They will be added here.

## Requirements
* [Numpy](http://www.numpy.org/)
* [Numba](http://numba.pydata.org/numba-doc/dev/index.html) is a jit compiler for python that we use to accelerate key code sections.

Although the package is sklearn-compatible and uses the sklearn estimator API, 
it does not depend on sklearn.

## Installation
This package is not on pypi, so you need to clone from github. Then you can install with pip.

    git clone https://github.com/davidthaler/arboretum.git
    cd arboretum
    pip install -e .

The `pip install -e .` installation allows you to hack on `arboretum` and still have your changes 
show up when you `import arboretum`.

## Usage
This package uses the sklearn estimator API. 
Classifiers have fit/predict/predict_proba methods.

    from arboretum import RFClassifier
    rf = RFCLassifier(n_trees=100)
    rf.fit(x_train, y_train)
    prob = rf.predict_proba(x_test)

Random forest always creates OOB predictions during fitting.
The OOB predictions are in a field of the fitted estimator:

    oob_prob = rf.oob_decision_function_

Regressors have only fit/predict, no predict_proba.

    from arboretum import RegressionTree
    regtree = RegressionTree(min_leaf=5)
    regtree.fit(x_train, y_train)
    pred = regtree.predict(x_test)

## More on Numba
Numba is a just-in-time compiler (jit) for python code.
It supports a only subset of standard python and the pydata stack (numpy/scipy/etc. ...), 
so it can't be used everywhere.
See numba's doc on [python suppport](http://numba.pydata.org/numba-doc/dev/reference/pysupported.html) for the standard language support
Unlike [pypy](https://pypy.org/), numba does support numpy arrays and a large number of numpy functions. 
See numba's doc on [numpy suppport](http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html) for what parts are/are not supported.
Support for recursion and OOP is experimental/unstable.
This project uses Numba to accelerate key sections of the code, like the splitter and apply.
Numba can be difficult to get installed. 
I recommend using the [Anaconda](https://docs.continuum.io/anaconda/) python distribution, which includes it.
