"""File with samplers that create points with some kind of ordered strcture.
"""
import numpy as np

from .sampler_base import PointSampler
from ..domains.domain1D import Interval


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
        points = np.linspace(0, 1, len(self)+2)[1:-1]
        if self.exponent > 1:
            points = points**self.exponent
        else:
            points = 1 - points**(1/self.exponent)
        length = self.domain.upper_bound - self.domain.lower_bound
        points = points * length + self.domain.lower_bound
        return  self.domain.space.embed(points.reshape(-1, 1))