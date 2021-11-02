import collections
import torch
import warnings

from .domain import Domain
from .newdomain0D import Point

class ProductDomain(Domain):

    def __init__(self, domain_a, domain_b):
        self.domain_a = domain_a
        self.domain_b = domain_b
        if not self.domain_a.space.keys().isdisjoint(self.domain_b.space):
            warnings.warn("""Warning: The space of a ProductDomain will be the product
                of its factor domains spaces. This may lead to unexpected behaviour.""")
        # check dependencies, so that at most domain_a needs variables of domain_b
        self._check_variable_dependencies()
        # set domain params
        space = self.domain_a.space * self.domain_b.space
        super().__init__(space=space, constructor=None, params={},
                         dim=domain_a.dim + domain_b.dim)
        
        # necessary variables consist of variables for both domains
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)
        
    def _check_variable_dependencies(self):
        a_variables_in_b = any(var in self.domain_b.necessary_variables for
                               var in self.domain_a.space)
        b_variables_in_a = any(var in self.domain_a.necessary_variables for
                               var in self.domain_b.space)
        name_a = self.domain_a.__class__.__name__
        name_b = self.domain_b.__class__.__name__
        if a_variables_in_b and b_variables_in_a:
            raise AssertionError(f"""Both domains {name_a}, {name_b} dependt on the 
                                     variables of the other domain. Will not be able 
                                     to resolve order of point creation!""")
        elif a_variables_in_b:
            raise AssertionError(f"""Domain_a: {name_b} depends on the variables of 
                                     domain_b: {name_a}, maybe you meant to use:
                                     domain_b * domain_a (multiplication
                                     is not commutative)""")

    def __call__(self, **data):
        # evaluate both domains at the given data 
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        # check if the data fixes a variable that would be computed with this domain:
        a_variables_in_data = all(k in data.keys() for k in self.domain_a.space.keys())
        b_variables_in_data = all(k in data.keys() for k in self.domain_b.space.keys())
        if a_variables_in_data: # domain_a will be a fixed point
            domain_a = self._create_point_from_data(self.domain_a, **data)
        if b_variables_in_data: # domain_b will be a fixed point 
            domain_b = self._create_point_from_data(self.domain_b, **data)
        return ProductDomain(domain_a=domain_a, domain_b=domain_b)

    def _create_point_from_data(self, domain, **data):
        point_data = []
        for vname in domain.space:
            if isinstance(data[vname], collections.Iterable):
                point_data.extend(data[vname])
            else:
                point_data.append(data[vname])
        return Point(space=domain.space, point=point_data)

    @property
    def boundary(self):
        # Domain object of the boundary
        boundary_1 = ProductDomain(self.domain_a.boundary, self.domain_b)
        boundary_2 = ProductDomain(self.domain_a, self.domain_b.boundary)
        return UnionDomain(boundary_1, boundary_2)

    @property
    def inner(self):
        # open domain
        return ProductDomain(self.domain_a.inner, self.domain_b.inner)

    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        assert self.dim == other.dim
        assert self.space == other.space
        raise NotImplementedError

    def __sub__(self, other):
        """Creates the cut of other from self.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to be of the same dimension.
        """
        return ProductDomain(self.domain_a & other, self.domain_b & other)

    def __contains__(self, points, **params):
        in_a = self.domain_a.__contains__(points, **params)
        in_b = self.domain_b.__contains__(points, **params)
        return torch.logical_and(in_a, in_b)

    def bounding_box(self, **params):
        bounds_a = self.domain_a.bounding_box(**params)
        bounds_b = self.domain_b.bounding_box(**params)
        bounds_a.extend(bounds_b)
        return bounds_a

    def sample_grid(self, n=None, d=None, **params):
        raise NotImplementedError

    def sample_random_uniform(self, n=None, d=None, **params):
        n, new_params = self._repeat_params(n, **params)
        b_points = self.domain_b.sample_random_uniform(n=n, d=d, **new_params) 
        a_points = self.domain_a.sample_random_uniform(n=1, d=d, **new_params, **b_points)
        return {**a_points, **b_points}

    def _repeat_params(self, n, **args):
        repeated_params = {}
        param_len = 1
        for key, domain_param in args.items():
            repeated_params[key] = torch.repeat_interleave(domain_param, n, dim=0)
            param_len = len(repeated_params[key])
        if param_len > 1:
            n = 1
        return n, repeated_params


"""
Classes for boolean domains
"""

class UnionDomain(Domain):
    
    def __init__(self, domain_a, domain_b):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        super().__init__(constructor=None, params={}, 
                         space=domain_a.space, dim=domain_a.dim)

    def __contains__(self, points, **params):
        in_a = self.domain_a.__contains__(points, **params)
        in_b = self.domain_b.__contains__(points, **params)
        return torch.logical_or(in_a, in_b)

    def __call__(self, **data):
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        return UnionDomain(domain_a, domain_b)

    def bounding_box(self, **params):
        bounds_a = self.domain_a.bounding_box(**params)
        bounds_b = self.domain_b.bounding_box(**params)
        bounds = []
        for i in range(self.space.dim):
            bounds.append(min([bounds_a[2*i], bounds_b[2*i]]))
            bounds.append(max([bounds_a[2*i+1], bounds_b[2*i+1]]))
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        points_a = self.domain_a.sample_random_uniform(n=n, d=d, **params)
        points_b = self.domain_b.sample_random_uniform(n=n, d=d, **params)
        in_a = self.domain_a.__contains__(points=points_b, **params)
        return super().sample_random_uniform(n=n, d=d, **params)

class CutDomain(Domain):
    pass


class IntersectionDomain(Domain):
    pass