import torch

from .domain import Domain


class Point(Domain):
    """Creates a single point at the given coordinates.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    coord : Number, List or callable
        The coordinate of the point.
    """
    def __init__(self, space, point):
        self.bounding_box_tol = 0.1
        params = {'point': point}
        super().__init__(constructor=Point, params=params,
                         space=space, dim=0)

    def __contains__(self, points, **params):
        point_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        inside = torch.isclose(points[:, None], point_params['point'])
        return torch.all(inside, dim=2)

    def bounding_box(self, **params):
        if callable(self.params['point']): # if point moves
             return self._bounds_for_callable_point(**params)
        if isinstance(self.params['point'], (torch.Tensor, list)):
             return self._bounds_for_higher_dimensions(**params)
        return [self.params['point'] - self.bounding_box_tol, 
                self.params['point'] + self.bounding_box_tol]

    def _bounds_for_callable_point(self, **params):
        point_params = self._domain_construction(**params)
        bounds = []
        discrete__points = point_params['point'].reshape(-1, self.space.dim)
        for i in range(self.space.dim):
            min_ = torch.min(discrete__points[:, i])
            max_ = torch.max(discrete__points[:, i])
            if min_ == max_:
                min_ -= self.bounding_box_tol
                max_ += self.bounding_box_tol
            bounds.append(min_.item()), bounds.append(max_.item())
        return bounds

    def _bounds_for_higher_dimensions(self):
        bounds = []
        for i in range(self.space.dim):
            p = self.params['point'][i]
            # substract/add a value to get a real bounding box, 
            # important if we later use these values to normalize the input
            bounds.append(p - self.bounding_box_tol)
            bounds.append(p + self.bounding_box_tol)
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        point_params = self._domain_construction(**params)
        points = torch.ones((point_params['param_len'], n, self.space.dim))
        points *= point_params['point']
        return self._divide_points_to_space_variables(points.reshape(-1, self.space.dim))

    def sample_grid(self, n=None, d=None, **params):
        # for one single point grid and random sampling is the same
        return self.sample_random_uniform(n=n, d=d, **params)