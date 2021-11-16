"""File with samplers that create random distributed points.
"""
import torch

from .sampler_base import PointSampler


class RandomUniformSampler(PointSampler):
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
        removed must return False. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """
    def __init__(self, domain, n_points=None, density=None, filter=None):
        super().__init__(n_points=n_points, density=density, filter=filter)
        self.domain = domain

    def _sample_points(self, **params):
        if self.n_points:
            rand_points = self.domain.sample_random_uniform(self.n_points, **params)
            repeated_params = self._repeat_input_params(self.n_points, **params)
            return {**rand_points, **repeated_params}
        else: # density is used
            sample_function = self.domain.sampler_random_uniform
            if any(var in self.domain.necessary_variables for var in params.keys()):
                return self._sample_params_dependent(sample_function, **params)
            return self._sample_params_independent(sample_function, **params)