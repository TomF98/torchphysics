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
        lower_bound, upper_bound = self.transform_to_user_functions(lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(space=space, dim=1)
        self.set_necessary_variables(self.lower_bound, self.upper_bound)

    def __call__(self, **data):
        new_lower_bound = self.lower_bound.partially_evaluate(**data)
        new_upper_bound = self.upper_bound.partially_evaluate(**data)
        return Interval(space=self.space, lower_bound=new_lower_bound, 
                        upper_bound=new_upper_bound)

    def __contains__(self, points, **params):
        lb = self.lower_bound(**points, **params)
        ub = self.upper_bound(**points, **params)
        points = self.space.as_tensor(points)
        bigger_then_low = torch.ge(points[:, None], lb) 
        smaller_then_up = torch.le(points[:, None], ub) 
        return torch.logical_and(bigger_then_low, smaller_then_up).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        lb = self.lower_bound(**params)
        ub = self.upper_bound(**params)
        points = torch.rand((self.get_num_of_params(**params), n, 1))
        points *= (ub - lb)
        points += lb
        return self.space.embed(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        lb = self.lower_bound(**params)
        ub = self.upper_bound(**params)
        points = torch.linspace(0, 1, n+2)[1:-1, None]
        points = (ub - lb) * points 
        points += lb
        return self.space.embed(points.reshape(-1, 1))

    def bounding_box(self, **params):
        lb = self.lower_bound(**params)
        ub = self.upper_bound(**params)
        return [torch.min(lb).item(), torch.max(ub).item()]

    def volume(self, **params):
        lb = self.lower_bound(**params)
        ub = self.upper_bound(**params)
        return (ub - lb).reshape(-1, 1)

    @property
    def boundary(self):
        return IntervalBoundary(self)

    @property
    def boundary_left(self):
        return IntervalSingleBoundaryPoint(self, side=self.lower_bound)

    @property
    def boundary_right(self):
        return IntervalSingleBoundaryPoint(self, side=self.upper_bound, normal_vec=1)


class IntervalBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Interval)
        super().__init__(domain)
    
    def __contains__(self, points, **params):
        close_to_left, close_to_right = self._check_close_left_right(points, params) 
        return torch.logical_or(close_to_left, close_to_right).reshape(-1, 1)

    def _check_close_left_right(self, points, params):
        lb = self.domain.lower_bound(**points, **params)
        ub = self.domain.upper_bound(**points, **params)
        points = self.space.as_tensor(points)
        close_to_left = torch.isclose(points[:, None], lb)
        close_to_right = torch.isclose(points[:, None], ub)
        return close_to_left, close_to_right

    def sample_random_uniform(self, n=None, d=None, **params):
        lb = self.domain.lower_bound(**params)
        ub = self.domain.upper_bound(**params)
        random_boundary_index = torch.rand((self.get_num_of_params(**params), n, 1)) < 0.5 
        points = torch.where(random_boundary_index, lb, ub)
        return self.space.embed(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        lb = self.domain.lower_bound(**params)
        ub = self.domain.upper_bound(**params)
        b_index = torch.tensor([0, 1], dtype=bool).repeat(int(n/2.0) + 1)
        points = torch.where(b_index[:n], lb, ub)
        return self.space.embed(points.reshape(-1, 1))

    def normal(self, points, **params):
        close_to_left, _ = self._check_close_left_right(points, params)
        return torch.where(close_to_left, -1, 1).reshape(-1, 1)

    def volume(self, **params):
        no_of_params = self.get_num_of_params(**params)
        return 2 * torch.ones((no_of_params, 1))


class IntervalSingleBoundaryPoint(BoundaryDomain):

    def __init__(self, domain, side, normal_vec=-1):
        assert isinstance(domain, Interval)
        super().__init__(domain)
        self.side = side
        self.normal_vec = normal_vec

    def __call__(self, **data):
        evaluate_domain = self.domain(**data)
        return IntervalSingleBoundaryPoint(evaluate_domain, side=self.side,
                                           normal_vec=self.normal_vec)

    def __contains__(self, points, **params):
        side = self.side(**points, **params)
        points = self.space.as_tensor(points)
        inside = torch.isclose(points[:, None], side)
        return inside.reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, **params):
        side = self.side(**params)
        points = torch.ones((self.get_num_of_params(**params), n, 1))
        points *= side
        return self.space.embed(points.reshape(-1, 1))

    def sample_grid(self, n=None, d=None, **params):
        return self.sample_random_uniform(n=n, d=d, **params)

    def normal(self, points, **params):
        points = torch.ones((self.get_num_of_params(**points, **params), 1))
        return points * self.normal_vec

    def volume(self, **params):
        no_of_params = self.get_num_of_params(**params)
        return 1 * torch.ones((no_of_params, 1))