"""File with samplers that create random distributed points.
"""
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
            sample_function = self.domain.sample_random_uniform
            if any(var in self.domain.necessary_variables for var in params.keys()):
                return self._sample_params_dependent(sample_function, **params)
            return self._sample_params_independent(sample_function, **params)

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
        sample_function = self.domain.sample_random_uniform
        num_of_params = self._extract_tensor_len_from_dict(params)
        point_dict = None
        for i in range(num_of_params):
            new_points_dict = {}
            num_of_new_points = 0
            iterations = 0
            # we have to make sure to sample for each params exactly n points
            while num_of_new_points < self.n_points:
                # sample points
                new_points, repeated_params = \
                    self._sample_for_ith_param(sample_function, params, i)
                new_points.update(repeated_params)
                # apply filter and save valid points
                num_of_new_points += self._apply_filter(new_points)
                new_points_dict =self._set_point_dict(new_points_dict, new_points)
                iterations += 1
                self._check_iteration_number(iterations, num_of_new_points)
            # if to many points were sampled, delete them.
            self._cut_tensor_to_length_n(new_points_dict)
            point_dict = self._set_point_dict(point_dict, new_points_dict)
        return point_dict 

    def _set_point_dict(self, point_dict, new_points_dict):
        if not point_dict:
            point_dict = new_points_dict
        else:
            self._append_point_dict(point_dict, new_points_dict)
        return point_dict