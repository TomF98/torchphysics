import torch
import numpy as np

from ..domain import Domain, BoundaryDomain


class Circle(Domain):
    """Class for circles.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the circle, e.g. center = [5,0].
    radius : number or callable
        The radius of the circle.
    """   
    def __init__(self, space, center, radius):
        assert space.dim == 2
        center, radius = self.transform_to_user_functions(center, radius)
        self.center = center
        self.radius = radius
        super().__init__(space=space, dim=2)
        self.set_necessary_variables(self.radius, self.center)

    def __call__(self, **data):
        new_center = self.center.partially_evaluate(**data)
        new_radius = self.radius.partially_evaluate(**data)
        return Circle(space=self.space, center=new_center, radius=new_radius)

    def _contains(self, points, **params):
        center, radius = self._compute_center_and_radius(**params, **points)
        points = self.space.as_tensor(points)
        norm = torch.linalg.norm(points - center, dim=1).reshape(-1, 1)
        return torch.le(norm[:, None], radius).reshape(-1, 1)

    def bounding_box(self, **params):
        center, radius = self._compute_center_and_radius(**params)
        bounds = []
        for i in range(self.dim):
            i_min = torch.min(center[:, i] - radius)
            i_max = torch.max(center[:, i] + radius)
            bounds.append(i_min.item())
            bounds.append(i_max.item())
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        if d:
            n = self.compute_n_from_density(d, **params)
        center, radius = self._compute_center_and_radius(**params)
        num_of_params = self.get_num_of_params(**params)
        r = torch.sqrt(torch.rand((num_of_params, n, 1)))
        r *= radius
        phi = 2 * np.pi * torch.rand((num_of_params, n, 1))
        points = torch.cat((torch.multiply(r, torch.cos(phi)),
                            torch.multiply(r, torch.sin(phi))), dim=2)
        # [:,None,:] is needed so that the correct entries will be added
        points += center[:, None, :]
        return self.space.embed(points.reshape(-1, 2))

    def sample_grid(self, n=None, d=None, **params):
        if d:
            n = self.compute_n_from_density(d, **params)
        center, radius = self._compute_center_and_radius(**params)
        num_of_params = self.get_num_of_params(**params)
        grid = self._equidistant_points_in_circle(n)
        grid = grid.repeat(num_of_params, 1).view(num_of_params, n, 2) 
        points = torch.multiply(radius, grid)
        points += center[:, None, :]
        return self.space.embed(points.reshape(-1, 2))

    def _compute_center_and_radius(self, **params):
        center = self.center(**params).reshape(-1, 2)
        radius = self.radius(**params)
        return center,radius

    def _equidistant_points_in_circle(self, n):
        # use a sunflower seed arrangement:
        # https://demonstrations.wolfram.com/SunflowerSeedArrangements/
        gr = (np.sqrt(5) + 1)/2.0 # golden ratio
        points = torch.arange(1, n+1)
        phi = (2 * np.pi / gr) * points
        radius = torch.sqrt(points - 0.5) / np.sqrt(n - 0.5) 
        points = torch.column_stack((torch.multiply(radius, torch.cos(phi)),
                                     torch.multiply(radius, torch.sin(phi))))
        return points                             

    def _get_volume(self, **params):
        radius = self.radius(**params)
        volume = np.pi * radius**2
        return volume.reshape(-1, 1)

    @property
    def boundary(self):
        return CircleBoundary(self)


class CircleBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Circle)
        super().__init__(domain)

    def _contains(self, points, **params):
        center, radius = self.domain._compute_center_and_radius(**params, **points)
        points = self.space.as_tensor(points)
        norm = torch.linalg.norm(points - center, dim=1).reshape(-1, 1)
        return torch.isclose(norm[:, None], radius).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        if d:
            n = self.compute_n_from_density(d, **params)
        center, radius = self.domain._compute_center_and_radius(**params)
        phi = 2 * np.pi * torch.rand((self.get_num_of_params(**params), n, 1))
        points = torch.cat((torch.multiply(radius, torch.cos(phi)),
                            torch.multiply(radius, torch.sin(phi))), 
                            dim=2)
        points += center[:, None, :]
        return self.space.embed(points.reshape(-1, 2))

    def sample_grid(self, n=None, d=None, **params):
        if d:
            n = self.compute_n_from_density(d, **params)
        center, radius = self.domain._compute_center_and_radius(**params)
        num_of_params = self.get_num_of_params(**params)
        grid = torch.linspace(0, 2*np.pi, n+1)[:-1] # last one would be double
        phi = grid.repeat(num_of_params).view(num_of_params, n, 1) 
        points = torch.cat((torch.multiply(radius, torch.cos(phi)),
                            torch.multiply(radius, torch.sin(phi))), 
                            dim=2)
        points += center[:, None, :]
        return self.space.embed(points.reshape(-1, 2))

    def normal(self, points, **params):
        center, radius = self.domain._compute_center_and_radius(**params, **points)
        points = self.space.as_tensor(points)
        normal = points - center
        return torch.divide(normal[:, None], radius).reshape(-1, 2)

    def _get_volume(self, **params):
        radius = self.domain.radius(**params)
        volume = 2 * np.pi * radius
        return volume.reshape(-1, 1)
