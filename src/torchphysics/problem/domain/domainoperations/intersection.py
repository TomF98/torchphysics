import warnings
import torch

from ..domain import Domain, BoundaryDomain


class IntersectionDomain(Domain):

    def __init__(self, domain_a: Domain, domain_b: Domain):
        assert domain_a.space == domain_b.space
        self.domain_a = domain_a
        self.domain_b = domain_b
        super().__init__(space=domain_a.space, dim=domain_a.dim)
        self.necessary_variables = domain_a.necessary_variables.copy()
        self.necessary_variables.update(domain_b.necessary_variables)

    def __call__(self, **data):
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        return IntersectionDomain(domain_a, domain_b)

    def _contains(self, points, **params):
        in_a = self.domain_a._contains(points, **params)
        in_b = self.domain_b._contains(points, **params)
        return torch.logical_and(in_a, in_b)

    def volume(self, **params):
        warnings.warn("""Exact volume of this intersection is not known,
                         will use the estimate: volume = domain_a.volume. This
                         can lead to not perfectly distributed 
                         points while sampling.""")
        return self.domain_a.volume(**params)

    def bounding_box(self, **params):
        bounds_a = self.domain_a.bounding_box(**params)
        bounds_b = self.domain_b.bounding_box(**params)
        bounds = []
        for i in range(self.space.dim):
            bounds.append(max([bounds_a[2*i], bounds_b[2*i]]))
            bounds.append(min([bounds_a[2*i+1], bounds_b[2*i+1]]))
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_random_with_d(d, **params)
        return self.space.embed(points)

    def _sample_random_with_d(self, d, **params):
        points_a = self.domain_a.sample_random_uniform(d=d, **params)
        points = self._cut_points(points_a, **params)
        return points

    def _cut_points(self, points, **params):
        # check which points are in domain b
        n = self.get_num_of_params(**points)
        _, repeated_params = self._repeat_params(n, **params)
        in_b = self.domain_b._contains(points=points, **repeated_params)    
        index = torch.where(in_b)[0]
        return self.space.as_tensor(points)[index]

    def sample_grid(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_grid_with_d(d, **params)
        return self.space.embed(points)

    def _sample_grid_with_d(self, d, **params):
        points_a = self.domain_a.sample_grid(d=d, **params)
        points = self._cut_points(points_a, **params)
        return points

    @property
    def boundary(self):
        return IntersectionBoundaryDomain(self)


class IntersectionBoundaryDomain(BoundaryDomain):

    def __init__(self, domain: IntersectionDomain):
        assert not isinstance(domain.domain_a, BoundaryDomain)
        assert not isinstance(domain.domain_b, BoundaryDomain)
        super().__init__(domain)

    def _contains(self, points, **params):
        in_a = self.domain.domain_a._contains(points, **params)
        in_b = self.domain.domain_b._contains(points, **params)
        on_a_bound = self.domain.domain_a.boundary._contains(points, **params)
        on_b_bound = self.domain.domain_b.boundary._contains(points, **params)
        on_a_part = torch.logical_and(on_a_bound, in_b)
        on_b_part = torch.logical_and(on_b_bound, in_a)
        return torch.logical_or(on_a_part, on_b_part)

    def volume(self, **params):
        warnings.warn("""Exact volume of this intersection-boundary is not known,
                         will use the estimate: volume = domain_a.volume. This
                         can lead to not perfectly distributed 
                         points while sampling.""")
        return self.domain.domain_a.volume(**params)

    def sample_random_uniform(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_random_with_d(d, **params)
        return self.space.embed(points)

    def _sample_random_with_d(self, d, **params):
        points_a = self.domain.domain_a.boundary.sample_random_uniform(d=d, **params)
        points_a = self._delete_outer_points(points_a, self.domain.domain_b, **params) 
        points_b = self.domain.domain_b.boundary.sample_random_uniform(d=d, **params)  
        points_b = self._delete_outer_points(points_b, self.domain.domain_a, **params)     
        return torch.cat((points_a, points_b), dim=0)     

    def _delete_outer_points(self, points, domain, **params):
        n = self.get_num_of_params(**points)
        _, repeated_params = self._repeat_params(n, **params)
        inside = domain._contains(points, **repeated_params)
        points = self.space.as_tensor(points)
        index = torch.where(inside)[0]
        return points[index]

    def sample_grid(self, n=None, d=None, **params):
        if n:
            raise NotImplementedError
        else:
            points = self._sample_grid_with_d(d, **params)
        return self.space.embed(points)

    def _sample_grid_with_d(self, d, **params):
        points_a = self.domain.domain_a.boundary.sample_grid(d=d, **params)
        points_a = self._delete_outer_points(points_a, self.domain.domain_b, **params) 
        points_b = self.domain.domain_b.boundary.sample_grid(d=d, **params)  
        points_b = self._delete_outer_points(points_b, self.domain.domain_a, **params)     
        return torch.cat((points_a, points_b), dim=0)     

    def normal(self, points, **params):
        a_normals = self.domain.domain_a.boundary.normal(points, **params)
        b_normals = self.domain.domain_b.boundary.normal(points, **params)
        on_a = self.domain.domain_a.boundary._contains(points, **params)
        normals = torch.where(on_a, a_normals, b_normals)
        return normals