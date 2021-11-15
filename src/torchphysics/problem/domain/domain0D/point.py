import torch

from ..domain import Domain


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
        point = self.transform_to_user_functions(point)[0]
        self.point = point
        super().__init__(space=space, dim=0)
        self.set_necessary_variables(point)

    def __call__(self, **data):
        new_point = self.point.partially_evaluate(**data)
        return Point(space=self.space, point=new_point)

    def _contains(self, points, **params):
        point_params = self.point(**params, **points)
        points = self.space.as_tensor(points)
        inside = torch.isclose(points[:, None], point_params)
        return torch.all(inside, dim=2)

    def bounding_box(self, **params):
        if callable(self.point.fun): # if point moves
             return self._bounds_for_callable_point(**params)
        if isinstance(self.point.fun, (torch.Tensor, list)):
             return self._bounds_for_higher_dimensions(**params)
        return [self.point.fun - self.bounding_box_tol, 
                self.point.fun + self.bounding_box_tol]

    def _bounds_for_callable_point(self, **params):
        bounds = []
        discrete__points = self.point(**params).reshape(-1, self.space.dim)
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
            p = self.point.fun[i]
            # substract/add a value to get a real bounding box, 
            # important if we later use these values to normalize the input
            bounds.append(p - self.bounding_box_tol)
            bounds.append(p + self.bounding_box_tol)
        return bounds

    def sample_random_uniform(self, n=None, d=None, **params):
        if d:
            n = self.compute_n_from_density(d, **params)
        point_params = self.point(**params)
        points = torch.ones((self.get_num_of_params(**params), n, self.space.dim))
        points *= point_params
        return self.space.embed(points.reshape(-1, self.space.dim))

    def sample_grid(self, n=None, d=None, **params):
        # for one single point grid and random sampling is the same
        return self.sample_random_uniform(n=n, d=d, **params)

    def _get_volume(self, **params):
        no_of_params = self.get_num_of_params(**params)
        return 1 * torch.ones((no_of_params, 1))