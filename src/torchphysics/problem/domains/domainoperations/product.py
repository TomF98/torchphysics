import collections
import torch
import warnings

from ..domain import BoundaryDomain, Domain
from ..newdomain0D import Point
from .union import UnionDomain
from ....utils.user_fun import UserFunction


N_APPROX_VOLUME = 10

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
        super().__init__(space=space, dim=domain_a.dim + domain_b.dim)
        
        # necessary variables consist of variables of both domains that are not given in domain_b
        self.necessary_variables \
            = (self.domain_a.necessary_variables - self.domain_b.space.variables) \
              | self.domain_b.necessary_variables

    def _check_variable_dependencies(self):
        a_variables_in_b = any(var in self.domain_b.necessary_variables for
                               var in self.domain_a.space)
        b_variables_in_a = any(var in self.domain_a.necessary_variables for
                               var in self.domain_b.space)
        name_a = self.domain_a.__class__.__name__
        name_b = self.domain_b.__class__.__name__
        if a_variables_in_b and b_variables_in_a:
            raise AssertionError(f"""Both domains {name_a}, {name_b} depend on the 
                                     variables of the other domain. Will not be able 
                                     to resolve order of point creation!""")
        elif a_variables_in_b:
            raise AssertionError(f"""Domain_a: {name_b} depends on the variables of 
                                     domain_b: {name_a}, maybe you meant to use:
                                     domain_b * domain_a (multiplication
                                     is not commutative)""")
        elif b_variables_in_a:
            self._is_constant = False
        else:
            self._is_constant = True

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

    def _contains(self, points, **params):
        in_a = self.domain_a._contains(points, **params)
        in_b = self.domain_b._contains(points, **params)
        return torch.logical_and(in_a, in_b)

    def bounding_box(self, **params):
        bounds_a = self.domain_a.bounding_box(**params)
        bounds_b = self.domain_b.bounding_box(**params)
        bounds_a.extend(bounds_b)
        return bounds_a
    
    def set_volume(self, volume):
        self._volume = self.transform_to_user_functions(volume)
    
    def _get_volume(self, **params):
        if self._is_constant:
            return self.domain_a.volume(**params) * self.domain_b.volume(**params)
        else:
            warnings.warn(f"""The volume of a ProductDomain where one factor domain depends on the
                              other can only be approximated by evaluating functions at {N_APPROX_VOLUME}
                              points. If you need exact volume or sampling, use domain.set_volume().""")
            # approximate the volume
            n, new_params = self._repeat_params(n=N_APPROX_VOLUME, **params)
            b_points = self.domain_b.sample_random_uniform(n=n, **new_params)
            if len(self.domain_b.necessary_variables) > 0:
                # points need to be sampled in every call to this function
                return sum(self.domain_a.volume(**b_points, **new_params))/N_APPROX_VOLUME \
                    * self.domain_b.volume(**params)
            elif len(self.necessary_variables) > 0:
                # we can keep the sampled points and evaluate domain_a in a function
                b_volume = self.domain_b.volume()
                def avg_volume(**local_params):
                    _, new_params = self._repeat_params(n=N_APPROX_VOLUME, **local_params)
                    return (sum(self.domain_a.volume(**b_points, **new_params))/N_APPROX_VOLUME \
                        * b_volume)
                args = self.domain_a.necessary_variables - self.domain_b.space.variables
                self._user_volume = UserFunction(avg_volume, args=args)
                return avg_volume(**params)
            else:
                # we can compute the volume only once and save it
                volume = sum((self.domain_a.volume(**b_points))/N_APPROX_VOLUME \
                    * self.domain_b.volume())
                self.set_volume(volume)
                return volume
            

    def sample_grid(self, n=None, d=None, **params):
        raise NotImplementedError(
            """Grid sampling on a product domain is not implmented. Use a product sampler
               instead.""")

    def sample_random_uniform(self, n=None, d=None, **params):
        if n:
            n, new_params = self._repeat_params(n, **params)
            b_points = self.domain_b.sample_random_uniform(n=n, **new_params) 
            a_points = self.domain_a.sample_random_uniform(n=1, **new_params, **b_points)
        else:
            assert d is not None
            n = int(d*self.volume())
            self.sample_random_uniform(n=n, **params)
        return {**a_points, **b_points}