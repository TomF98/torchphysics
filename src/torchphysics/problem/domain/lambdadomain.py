import torch
import copy 

from .domain import Domain, ProductDomain
from ...utils.user_fun import UserFunction


class LambdaDomain(Domain):
    def __init__(self, constructor, params, space, dim):
        """
        params : A dictionary containing all the params needed to create the given
            domain_class domain
        """
        super().__init__(space, dim=dim)
        self.constructor = constructor
        self.params = params

        # create a set of variables/spaces that this domain needs to be properly defined
        self.necessary_variables = set()
        for key in self.params:
            if callable(self.params[key]):
                self.params[key] = UserFunction(params[key])
                for k in self.params[key].necessary_args:
                    self.necessary_variables.add(k)
        assert not any(var in self.necessary_variables for var in self.space)

    def __call__(self, **data):
        """
        (Partially) evaluate given lambda functions.
        """
        evaluated_params = {}
        for key in self.params:
            evaluated_params[key] = self._call_param(self.params[key], data)
        if all(var in data for var in self.necessary_variables):
            return self.constructor(space=self.space, **evaluated_params)
        else:
            self.params = evaluated_params
            return self

    def _call_param(self, param, args):
        if callable(param):
            if all(arg in args for arg in param.necessary_args):
                return param(**args)
            else:
                # to avoid manipulation of given param obj, we create a copy
                copy.deepcopy(param).set_default(**args)
        return param

    def _domain_construction(self, **params):
        d_params = {'param_len': 1}
        for key, domain_param in self.params.items():
            if callable(domain_param):
                d_params[key] = domain_param(**params)[:, None] 
                d_params['param_len'] = d_params[key].shape[0]
            else:
                if isinstance(domain_param, torch.Tensor):
                    d_params[key] = domain_param
                else: 
                    d_params[key] = torch.tensor(domain_param)
        return d_params

    def __mul__(self, other):
        pass
        #return LambdaProductDomain(self, other)


"""
class LambdaDomain(Domain):
    def __init__(self, constructor, params, space, dim):
        ""
        params : A dictionary containing all the params needed to create the given
            domain_class domain
        ""
        super().__init__(space, dim=dim)
        self.constructor = constructor
        self.params = params

        # create a set of variables/spaces that this domain needs to be properly defined
        self.necessary_variables = set()
        for key in self.params:
            if callable(self.params[key]):
                self.params[key] = UserFunction(params[key])
                for k in self.params[key].necessary_args:
                    self.necessary_variables.add(k)
        assert not any(var in self.necessary_variables for var in self.space)

    def __call__(self, **data):
        ""
        (Partially) evaluate given lambda functions.
        ""
        evaluated_params = {}
        for key in self.params:
            evaluated_params[key] = self._call_param(self.params[key], data)
        if all(var in data for var in self.necessary_variables):
            return self.constructor(**evaluated_params)
        else:
            return LambdaDomain(constructor=self.constructor,
                                params=evaluated_params,
                                space=self.space,
                                dim=self.dim)

    def _call_param(self, param, args):
        if callable(param):
            if all(arg in args for arg in param.necessary_args):
                return param(**args)
            else:
                # to avoid manipulation of given param obj, we create a copy
                copy.deepcopy(param).set_default(**args)
        return param

    def __mul__(self, other):
        pass
        #return LambdaProductDomain(self, other)

class LambdaProductDomain(ProductDomain):

    def __new__(cls, domain_a, domain_b):
        # case handling:
        if isinstance(domain_a, LambdaDomain) and isinstance(domain_b, LambdaDomain):
            if any(var in domain_a.necessary_variables for var in domain_b.space):
                if any(var in domain_b.necessary_variables for var in domain_a.space):
                    raise ValueError("Dependencies of LambdaDomain should always be
                        in a single direction. Please define domains succesively.")
                else:  # domain_a is build upon domain_b
                    
        if not isinstance(other, LambdaDomain) and all(var in other.space for var in self.necessary_variables):
            return LambdaProductDomain(self, other)
        if isinstance(other, LambdaDomain):
            return LambdaDomain(constructor=constr, params=)

        return super().__new__()

    def __init__(self, domain_a, domain_b):
        super().__init__()
        if isinstance(self, LambdaDomain):

    def is_inside(self, points):
        return super().is_inside(points)

    def bounding_box(self):
        # NOTE: This is only an estimate on the bounding box
        return super().bounding_box()

    def __add__(self, other):
        pass

    def __or__(self, other):
        # Union
        return self.__add__(other)

    def __sub__(self, other):
        # Difference
        pass

    def __and__(self, other):
        # Intersection
        pass
"""