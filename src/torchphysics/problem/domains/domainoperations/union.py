import warnings
import torch

from ..domain import Domain, BoundaryDomain


class UnionDomain(Domain):
    
    def __init__(self, domain_a: Domain, domain_b: Domain, disjoint=False):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.disjoint = disjoint
        super().__init__(space=domain_a.space, dim=domain_a.dim)
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)

    def volume(self, return_value_of_a_b=False, **params):
        if not self.disjoint:
            warnings.warn("""Exact volume of this union is not known, will use the
                             estimate: volume = domain_a.volume + domain_b.volume.
                             If you need exact volume or sampling, use domain.set_volume().""")
        volume_a = self.domain_a.volume(**params)
        volume_b = self.domain_b.volume(**params)
        if return_value_of_a_b:
            return volume_a + volume_b, volume_a, volume_b
        return volume_a + volume_b

    def _contains(self, points, **params):
        in_a = self.domain_a._contains(points, **params)
        in_b = self.domain_b._contains(points, **params)
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
        if n:
            points = self._sample_random_with_n(n, **params)
        else: # d not None
            points = self._sample_random_with_d(d, **params)
        return self.space.embed(points)

    def _sample_random_with_n(self, n, **params):
        # sample n points in both domains
        points_a = self.domain_a.sample_random_uniform(n=n, **params)
        points_b = self.domain_b.sample_random_uniform(n=n, **params)
        # check which points of domain b are in domain a
        _, repeated_params = self._repeat_params(n, **params)
        in_a = self.domain_a._contains(points=points_b, **repeated_params)
        # approximate volume of this domain
        volume_approx, volume_a, _ = self.volume(return_value_of_a_b=True,
                                                 **repeated_params)
        volume_ratio = torch.divide(volume_a, volume_approx)
        # choose points depending of the proportion of the domain w.r.t. the
        # whole domain union
        rand_index = torch.rand((max(n, self.get_num_of_params(**repeated_params)), 1))
        rand_index = torch.logical_or(in_a, rand_index <= volume_ratio)
        points = torch.where(rand_index,
            	             self.space.as_tensor(points_a),
                             self.space.as_tensor(points_b))                
        return points

    def _sample_random_with_d(self, d, **params):
        # sample n points in both domains
        points_a = self.domain_a.sample_random_uniform(d=d, **params)
        points_b = self.domain_b.sample_random_uniform(d=d, **params)      
        points = self._append_points(points_a, points_b, **params)
        return points

    def _append_points(self, points_a, points_b, **params):
        in_a = self._points_lay_in_other_domain(points_b, self.domain_a, **params)  
        # delete the points that are in domain a (so the sampling stays uniform)
        index = torch.where(torch.logical_not(in_a))[0]
        points = torch.cat((self.space.as_tensor(points_a), 
                            self.space.as_tensor(points_b)[index]), dim=0)         
        return points

    def _points_lay_in_other_domain(self, points, domain, **params):
        # check which points of domain b are in domain a
        n = self.get_num_of_params(**points)
        _, repeated_params = self._repeat_params(n, **params)
        in_a = domain._contains(points=points, **repeated_params)
        return in_a

    def sample_grid(self, n=None, d=None, **params):
        if n:
            points = self._sample_grid_with_n(n, **params)
        else: # d not None
            points = self._sample_grid_with_d(d, **params)
        return points

    def _sample_grid_with_n(self, n, **params):
        volume_approx, volume_a, _ = self.volume(return_value_of_a_b=True,
                                                 **params)
        scaled_n = int(torch.ceil(n * volume_a/volume_approx))
        points_a = self.domain_a.sample_grid(n=scaled_n, **params)
        if n - scaled_n > 0:
            self._sample_in_b(n, params, points_a)
        return points_a

    def _sample_in_b(self, n, params, points_a):
        # check how many points from domain a lay in b, these points will not be used!
        in_b = self._points_lay_in_other_domain(points_a, self.domain_b, **params)
        index = torch.where(torch.logical_not(in_b))[0]
        scaled_n = n - len(index)
        points_b = self.domain_b.sample_grid(n=scaled_n, **params)
        for key in self.space:
            points_a[key] = torch.cat((points_a[key][index], points_b[key]), dim=0)

    def _sample_grid_with_d(self, d, **params):
        points_a = self.domain_a.sample_grid(d=d, **params)
        points_b = self.domain_b.sample_grid(d=d, **params)      
        points = self._append_points(points_a, points_b, **params)
        return self.space.embed(points)

    @property
    def boundary(self):
        return UnionBoundaryDomain(self)


class UnionBoundaryDomain(BoundaryDomain):

    def __init__(self, domain: UnionDomain):
        assert not isinstance(domain.domain_a, BoundaryDomain)
        assert not isinstance(domain.domain_b, BoundaryDomain)
        super().__init__(domain)

    def _contains(self, points, **params):
        in_a = self.domain.domain_a._contains(points, **params)
        in_b = self.domain.domain_b._contains(points, **params)
        on_a_bound = self.domain.domain_a.boundary._contains(points, **params)
        on_b_bound = self.domain.domain_b.boundary._contains(points, **params)
        on_a_part = torch.logical_and(on_a_bound, torch.logical_not(in_b))
        on_b_part = torch.logical_and(on_b_bound, torch.logical_not(in_a))
        return torch.logical_or(on_a_part, on_b_part)

    def volume(self, **params):
        if not self.domain.disjoint:
            warnings.warn("""Exact volume of this domain is not known, will use the
                             estimate: volume = domain_a.volume + domain_b.volume.
                             If you need exact volume or sampling, use domain.set_volume().""")
        volume_a = self.domain.domain_a.boundary.volume(**params)
        volume_b = self.domain.domain_b.boundary.volume(**params)
        return volume_a + volume_b
    
    def sample_random_uniform(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_random_with_d(d, **params)
        return self.space.embed(points)

    def _sample_random_with_d(self, d, **params):
        points_a = self.domain.domain_a.boundary.sample_random_uniform(d=d, **params)
        points_a = self._delete_inner_points(points_a, self.domain.domain_b, **params)
        points_b = self.domain.domain_b.boundary.sample_random_uniform(d=d, **params)  
        points_b = self._delete_inner_points(points_b, self.domain.domain_a, **params)     
        return torch.cat((points_a, points_b), dim=0)     

    def _delete_inner_points(self, points, domain, **params):
        n = self.get_num_of_params(**points)
        _, repeated_params = self._repeat_params(n, **params)
        inside = domain._contains(points, **repeated_params)
        points = self.space.as_tensor(points)
        index = torch.where(torch.logical_not(inside))[0]
        return points[index]

    def sample_grid(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_grid_with_d(d, **params)
        return self.space.embed(points)

    def _sample_grid_with_d(self, d, **params):
        points_a = self.domain.domain_a.boundary.sample_grid(d=d, **params)
        points_a = self._delete_inner_points(points_a, self.domain.domain_b, **params)
        points_b = self.domain.domain_b.boundary.sample_grid(d=d, **params)  
        points_b = self._delete_inner_points(points_b, self.domain.domain_a, **params)     
        return torch.cat((points_a, points_b), dim=0)    

    def normal(self, points, **params):
        a_normals = self.domain.domain_a.boundary.normal(points, **params)
        b_normals = self.domain.domain_b.boundary.normal(points, **params)
        on_a = self.domain.domain_a.boundary._contains(points, **params)
        normals = torch.where(on_a, a_normals, b_normals)
        return normals