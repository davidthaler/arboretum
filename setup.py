from setuptools import setup

setup(
    name='arboretum',
    version='0.1.0',
    description='Python decision trees and ensembles, accelerated with numba',
    url='https://github.com/davidthaler/arboretum',
    author='David Thaler',
    author_email='davidthaler@gmail.com',
    license='MIT',
    packages=['arboretum', 'arboretum.datasets', 'arboretum.tests']
)
