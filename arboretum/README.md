# Arboretum
A collection of rare and exotic trees (and some ordinary ones).

## Requirements
* [Numba](http://numba.pydata.org/numba-doc/dev/index.html) Numba is a jit compiler for python that we use to accelerate key code sections.

## Installation
This package is not on pypi, so you need to clone from github. Then you can install with pip.

    git clone https://github.com/davidthaler/arboretum.git
    cd arboretum
    pip install -e .

The `pip install -e .` installation allows you to hack on `arboretum` and still have your changes 
show up when you `import arboretum`.

## Numba
Numba is a just-in-time compiler (jit) for python code.
It supports a only subset of standard python and the pydata stack (numpy/scipy/etc. ...), 
so it can't be used everywhere.
See numba's doc on [python suppport](http://numba.pydata.org/numba-doc/dev/reference/pysupported.html) for the standard language support
Unlike [pypy](https://pypy.org/), numba does support numpy arrays and a large number of numpy functions. 
See numba's doc on [numpy suppport](http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html) for what parts are/are not supported.
Support for recursion and OOP is experimental/unstable.
This project uses it to accelerate key sections of the code, like the splitter.
Numba can be difficult to get installed. 
I recommend using the [Anaconda](https://docs.continuum.io/anaconda/) python distribution, which includes it.
