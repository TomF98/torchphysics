import torch
import numpy as np

from .domain import Domain, BoundaryDomain


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
        params = {'center': center, 'radius': radius}
        super().__init__(space=space, dim=2, constructor=Circle, params=params)

    def _domain_construction(self, **params):
        d_params = super()._domain_construction(**params)
        d_params['center'] = d_params['center'].reshape(-1, 2)
        return d_params

    def __contains__(self, points, **params):
        circle_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        norm = torch.norm(points - circle_params['center'], dim=1).reshape(-1, 1)
        return torch.le(norm[:, None], circle_params['radius']).reshape(-1, 1)

    def bounding_box(self, **params):
        circle_params = self._domain_construction(**params)
        bounds = []
        for i in range(self.dim):
            i_min = torch.min(circle_params['center'][:, i] - circle_params['radius'])
            i_max = torch.max(circle_params['center'][:, i] + circle_params['radius'])
            bounds.append(i_min.item())
            bounds.append(i_max.item())
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        circle_params = self._domain_construction(**params)
        r = torch.sqrt(torch.rand((circle_params['param_len'], n, 1)))
        r *= circle_params['radius'] 
        phi = 2 * np.pi * torch.rand((circle_params['param_len'], n, 1))
        points = torch.cat((torch.multiply(r, torch.cos(phi)),
                            torch.multiply(r, torch.sin(phi))), dim=2)
        # [:,None,:] is needed so that the correct entries will be added
        points += circle_params['center'][:, None, :]
        return super()._divide_points_to_space_variables(points.reshape(-1, 2))

    def sample_grid(self, n=None, d=None, **params):
        circle_params = self._domain_construction(**params)
        grid = self._equidistant_points_in_circle(n)
        grid = grid.repeat(circle_params['param_len'], 1).view(circle_params['param_len'], n, 2) 
        points = torch.multiply(circle_params['radius'], grid)
        points += circle_params['center'][:, None, :]
        return super()._divide_points_to_space_variables(points.reshape(-1, 2))

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

    @property
    def boundary(self):
        return CircleBoundary(self)

class CircleBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Circle)
        super().__init__(domain)

    def __contains__(self, points, **params):
        circle_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        norm = torch.norm(points - circle_params['center'], dim=1).reshape(-1, 1)
        return torch.isclose(norm[:, None], circle_params['radius']).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        circle_params = self._domain_construction(**params)
        phi = 2 * np.pi * torch.rand((circle_params['param_len'], n, 1))
        points = torch.cat((torch.multiply(circle_params['radius'], torch.cos(phi)),
                            torch.multiply(circle_params['radius'], torch.sin(phi))), 
                            dim=2)
        points += circle_params['center'][:, None, :]
        return super()._divide_points_to_space_variables(points.reshape(-1, 2))

    def sample_grid(self, n=None, d=None, **params):
        circle_params = self._domain_construction(**params)
        grid = torch.linspace(0, 2*np.pi, n+1)[:-1] # last one would be double
        phi = grid.repeat(circle_params['param_len']).view(circle_params['param_len'], n, 1) 
        points = torch.cat((torch.multiply(circle_params['radius'], torch.cos(phi)),
                            torch.multiply(circle_params['radius'], torch.sin(phi))), 
                            dim=2)
        points += circle_params['center'][:, None, :]
        return super()._divide_points_to_space_variables(points.reshape(-1, 2))

    def normal(self, points, **params):
        circle_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        normal = points - circle_params['center']
        return torch.divide(normal[:, None], circle_params['radius']).reshape(-1, 2)