'''
Some functionality common to all arboretum models: get_params/
set_params, which are needed for compatability with sklearn.model_selection
and __repr__. 

This class requires its subclasses to follow the convention from sklearn that 
all estimator configuration paramters are keyword arguments of the constructor.
That allows cloning "model" like this:

    model.__class__(**model.get_params())

It also expects subclasses to set the n_features_ attribute in fit().

author: David Thaler
date: Septmenber 2017
'''
import inspect
import numpy as np


class NotImplemented(Exception):
    '''Exception to signal unimplemented superclass method called'''
    pass

class NotFitted(Exception):
    '''Exception to signal predict method called on unfitted estimator'''
    pass

class BaseModel:

    def __init__(self):
        '''Need to prevent calling through to object for get_params to work.'''
        pass

    def get_params(self):
        '''
        Generate a dict of all configuration params used by this model.

        Returns:
            dict of configuration params and their values
        '''
        sig = inspect.signature(self.__init__)
        names = list(sig.parameters.keys())
        params = {n:getattr(self, n) for n in names}
        return params

    def __repr__(self):
        '''
        Output model name and configuration (constructor params).

        Returns:
            string representation of this object
        '''
        name  = self.__class__.__name__
        params = self.get_params()
        params_list = ['{}={}'.format(*t) for t in params.items()]
        params_str = ', '.join(params_list)
        return '%s(%s)' % (name, params_str)

    def _predict_check(self, x):
        '''
        Checks that test data has correct type (numpy array), shape (2-D)
        and size (same column dimension as training data).1
        Also checks that this estimator is fitted.

        Args:
            x: Test data; 2-D ndarray with shape n_features columns

        Returns:
            nothing

        Raises:
            ValueError if x is wrong type, shape or size
            NotFitted if this estimator is not fitted.
        '''
        if type(x) is not np.ndarray:
            raise ValueError('Estimator requires numpy array input')
        if x.ndim != 2:
            raise ValueError('Prediction input must be 2-D array')
        if not self.is_fitted:
            raise NotFitted('Predict called before fit.')
        if x.shape[1] != self.n_features_:
            raise ValueError('Estimator fitted for %d-dimensions, but data has %d'
                                % (self.n_features_, x.shape[1]))

    @property
    def is_fitted(self):
        '''
        True/False if this estimator is fitted/not fitted.
        Fitted estimators must set the n_features_ attribute.

        Returns:
            True if this estimator has been fitted, False otherwise
        '''
        return hasattr(self, 'n_features_')
