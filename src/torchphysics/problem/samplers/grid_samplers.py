"""File with samplers that create points with some kind of ordered structure.
"""
import torch
import warnings

from .sampler_base import PointSampler
from ..domains.domain1D import Interval
from .random_samplers import RandomUniformSampler


class GridSampler(PointSampler):
    """Will sample random uniform distributed points in the given domain.

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desiered density of the created points.
    filter : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return false. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """
    def __init__(self, domain, n_points=None, density=None, filter=None):
        super().__init__(n_points=n_points, density=density, filter=filter)
        self.domain = domain

    def _sample_points(self, **params):
        if any(var in self.domain.necessary_variables for var in params.keys()):
            return self._sample_params_dependent(self.domain.sample_grid, **params)
        return self._sample_params_independent(self.domain.sample_grid, **params)

    def _sample_points_with_filter(self, **params):
        if self.n_points:
            point_dict = self._sample_n_points_with_filter(**params)
        else:
            # for density sampling, just sample normally and afterwards remove all 
            # points that are not allowed
            point_dict = self._sample_points(**params)
            _ = self._apply_filter(point_dict)
        return point_dict

    def _sample_n_points_with_filter(self, **params):
        # The idea is to first sample normally, then see how many points are valid.
        # Then rescale the number of points to get a better grid and sample again.
        # If still some points are missing add random points.
        sample_function = self.domain.sample_grid
        num_of_params = self._extract_tensor_len_from_dict(params)
        point_dict = None
        for i in range(num_of_params):
            ith_params, new_points, num_of_valid_points = \
                self._sample_grid(params, sample_function, i, self.n_points)
            new_points_dict = self._resample_grid(new_points, num_of_valid_points, 
                                                  sample_function, ith_params)
            # if to many points were sampled, delete the last ones.
            self._cut_tensor_to_length_n(new_points_dict)
            point_dict = RandomUniformSampler._set_point_dict(self,
                                                              point_dict,
                                                              new_points_dict)
        return point_dict 

    def _sample_grid(self, params, sample_function, i, n):
        ith_params = self._extract_points_from_dict(i, params)
        new_points = sample_function(n, **ith_params)
        num_of_points = self._extract_tensor_len_from_dict(new_points)
        repeated_params = self._repeat_input_params(num_of_points, **ith_params)
        new_points.update(repeated_params)
        num_of_valid_points = self._apply_filter(new_points)
        return ith_params, new_points, num_of_valid_points

    def _resample_grid(self, new_points, num_of_valid_points, sample_func,
                       current_params):
        if num_of_valid_points == self.n_points:
            # the first grid is already perfect
            return new_points
        elif num_of_valid_points == 0:
            warnings.warn("""First iteration did not find any valid grid points.
                             Will try again with n = 10 * self.n_points""")
            scaled_n = int(10*self.n_points)
        else:
            scaled_n = int(self.n_points**2/num_of_valid_points)
        _, new_points, num_of_valid_points = \
            self._sample_grid(current_params, sample_func, 0, scaled_n)
        self._append_random_points(new_points, num_of_valid_points, current_params)
        return new_points

    def _append_random_points(self, new_points, num_of_valid_points, current_params):
        if num_of_valid_points == self.n_points:
            return 
        random_sampler = RandomUniformSampler(domain=self.domain,
                                              n_points=self.n_points)
        random_sampler.filter = self.filter
        random_points = random_sampler.sample_points(**current_params)
        self._append_point_dict(new_points, random_points)
                            

class SpacedGridSampler(PointSampler):
    """Will sample non equdistant grid points in the given interval.
    This works only on intervals!

    Parameters
    ----------
    domain : Interval
        The Interval in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    exponent : Number
        Determines how non equdistant the points are and at which corner they
        are accumulated. They are computed with a grid in [0, 1]
        and then transformed with the exponent and later scaled/translated:
            exponent < 1: More points at the upper bound. 
                          points = 1 - x**(1/exponent)
            exponent > 1: More points at the lower bound.
                          points = x**(exponent)
    """
    def __init__(self, domain, n_points, exponent):
        assert isinstance(domain, Interval), """The domain has to be a interval!"""
        super().__init__(n_points=n_points)
        self.domain = domain
        self.exponent = exponent

    def sample_points(self, **params):
        if any(var in self.domain.necessary_variables for var in params.keys()):
            return self._sample_params_dependent(self._sample_spaced_grid, **params)
        return self._sample_params_independent(self._sample_spaced_grid, **params)

    def _sample_spaced_grid(self, _n=None, _d=None, **params):
        lb = self.domain.lower_bound(**params)
        ub = self.domain.upper_bound(**params)
        points = torch.linspace(0, 1, len(self)+2)[1:-1]
        if self.exponent > 1:
            points = points**self.exponent
        else:
            points = 1 - points**(1/self.exponent)
        interval_length = ub - lb
        points = points * interval_length + lb
        return  self.domain.space.embed(points.reshape(-1, 1))