import torch

from .domain import Domain, BoundaryDomain


class Interval(Domain):
    """Creates a Interval of the form [a, b].

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    lower_bound : Number or callable
        The left/lower bound of the interval.
    upper_bound : Number or callable
        The right/upper bound of the interval.
    """
    def __init__(self, space, lower_bound, upper_bound):
        assert space.dim == 1
        params = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
        super().__init__(space=space, dim=1, constructor=Interval, params=params)

    def _domain_construction(self, **params):
        d_params = super()._domain_construction(**params)
        d_params['length'] = d_params['upper_bound'] - d_params['lower_bound']
        return d_params

    def __contains__(self, points, **params):
        interval_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        bigger_then_low = torch.ge(points[:, None], interval_params['lower_bound']) 
        smaller_then_up = torch.le(points[:, None], interval_params['upper_bound']) 
        return torch.logical_and(bigger_then_low, smaller_then_up).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        interval_params = self._domain_construction(**params) 
        points = torch.rand((interval_params['param_len'], n, 1))
        points *= interval_params['length'] 
        points += interval_params['lower_bound']
        return super()._divide_points_to_space_variables(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        interval_params = self._domain_construction(**params) 
        points = torch.linspace(0, 1, n+2)[1:-1, None]
        points = interval_params['length'] * points 
        points += interval_params['lower_bound']
        return super()._divide_points_to_space_variables(points.reshape(-1, 1))

    def bounding_box(self, **params):
        interval_params = self._domain_construction(**params) 
        return [torch.min(interval_params['lower_bound']).item(), 
                torch.max(interval_params['upper_bound']).item()]

    @property
    def boundary(self):
        return IntervalBoundary(self)

    @property
    def boundary_left(self):
        return IntervalSingleBoundaryPoint(self, side='lower_bound')

    @property
    def boundary_right(self):
        return IntervalSingleBoundaryPoint(self, side='upper_bound')


class IntervalBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Interval)
        super().__init__(domain)
    
    def __contains__(self, points, **params):
        close_to_left, close_to_right = self._check_close_left_right(points, params) 
        return torch.logical_or(close_to_left, close_to_right).reshape(-1, 1)

    def _check_close_left_right(self, points, params):
        interval_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        close_to_left = torch.isclose(points[:, None], interval_params['lower_bound'])
        close_to_right = torch.isclose(points[:, None], interval_params['upper_bound'])
        return close_to_left, close_to_right

    def sample_random_uniform(self, n=None, d=None, **params):
        interval_params = self._domain_construction(**params)
        random_boundary_index = torch.rand((interval_params['param_len'], n, 1)) < 0.5 
        points = torch.where(random_boundary_index, interval_params['lower_bound'], 
                             interval_params['upper_bound'])
        return super()._divide_points_to_space_variables(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        interval_params = self._domain_construction(**params)
        b_index = torch.tensor([0, 1], dtype=bool).repeat(int(n/2.0) + 1)
        points = torch.where(b_index[:n], interval_params['lower_bound'], 
                             interval_params['upper_bound'])
        return super()._divide_points_to_space_variables(points.reshape(-1, 1))

    def normal(self, points, **params):
        close_to_left, _ = self._check_close_left_right(points, params)
        return torch.where(close_to_left, -1, 1).reshape(-1, 1)


class IntervalSingleBoundaryPoint(BoundaryDomain):

    def __init__(self, domain, side):
        assert isinstance(domain, Interval)
        super().__init__(domain)
        self.side = side

    def __call__(self, **data):
        evaluate_domain = self.domain(**data)
        return IntervalSingleBoundaryPoint(evaluate_domain, side=self.side)

    def __contains__(self, points, **params):
        interval_params = self._domain_construction(**params, **points)
        points = self._return_space_variables_to_point_list(points)
        inside = torch.isclose(points[:, None], interval_params[self.side])
        return inside.reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        interval_params = self._domain_construction(**params)
        points = torch.ones((interval_params['param_len'], n, 1))
        points *= interval_params[self.side]
        return self._divide_points_to_space_variables(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        return self.sample_random_uniform(n=n, d=d, **params)

    def normal(self, points, **params):
        interval_params = self._domain_construction(**params, **points)
        points = torch.ones((interval_params['param_len'], 1))
        if self.side == 'lower_bound':
            points *= -1
        return points