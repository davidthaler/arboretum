'''
Some functionality common to all arboretum models: get_params/
set_params, which are needed for compatability with sklearn.model_selection
and __repr__.

author: David Thaler
date: Septmenber 2017
'''
import inspect


class BaseModel:

    def __init__(self):
        '''Need to prevent calling through to object for get_params to work.'''
        pass

    def get_params(self):
        sig = inspect.signature(self.__init__)
        names = list(sig.parameters.keys())
        params = {n:getattr(self, n) for n in names}
        return params

    def __repr__(self):
        name  = self.__class__.__name__
        params = self.get_params()
        params_list = ['{}={}'.format(*t) for t in params.items()]
        params_str = ', '.join(params_list)
        return '%s(%s)' % (name, params_str)
