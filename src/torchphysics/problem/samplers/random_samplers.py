"""File with samplers that create random distributed points.
"""
import torch
import numbers

from .sampler_base import PointSampler
from ..domains.domain import BoundaryDomain
from ..spaces import Points


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
                new_points_dict = self._set_point_dict(new_points_dict, new_points)
                iterations += 1
                self._check_iteration_number(iterations, num_of_new_points)
            # if to many points were sampled, delete them.
            self._cut_tensor_to_length_n(new_points_dict)
            point_dict = self._set_point_dict(point_dict, new_points_dict)
        return point_dict 


class GaussianSampler(PointSampler):
    """Will sample normal/gaussian distributed points in the given domain.
    Only works for the inner part of a domain, not the boundary!

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    mean : list, array or tensor
        The center/mean of the distribution. Has to fit the dimension
        of the given domain.
    std : number
        The standard deviation of the distribution.
    """
    def __init__(self, domain, n_points, mean, std):
        assert not isinstance(domain, BoundaryDomain), \
            """Gaussian sampling is not implemented for boundaries."""
        super().__init__(n_points=n_points)
        self.domain = domain
        self.mean = mean
        self.std = torch.tensor(std)
        self._check_mean_correct_dim()

    def _check_mean_correct_dim(self):
        if isinstance(self.mean, numbers.Number):
            self.mean = torch.FloatTensor([self.mean])
        elif not isinstance(self.mean, torch.Tensor):
            self.mean = torch.FloatTensor(self.mean)
        assert len(self.mean) == self.domain.dim, \
            f"""Dimension of mean: {self.mean}, does not fit the domain.""" 

    def _sample_points(self, **params):
        num_of_params = self._extract_tensor_len_from_dict(params)
        point_dict = None
        torch_dis = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)
        for i in range(num_of_params):
            current_num_of_points = 0
            new_points_dict = {}
            while current_num_of_points < self.n_points:
                new_points = torch_dis.sample((self.n_points,))
                new_points = Points(new_points, self.domain.space)
                ith_params = self._extract_points_from_dict(i, params)
                repeat_params = self._repeat_input_params(self.n_points, **ith_params)
                new_points.update(repeat_params)
                current_num_of_points += self._check_inside_domain(new_points)
                new_points_dict = self._set_point_dict(new_points_dict, new_points)
            # if to many points were sampled, delete them.
            self._cut_tensor_to_length_n(new_points_dict)
            point_dict = self._set_point_dict(point_dict, new_points_dict)
        return point_dict

    def _check_inside_domain(self, new_points):
        inside = self.domain._contains(new_points)
        index = torch.where(inside)[0]
        for key, data in new_points.items():
            new_points[key] = data[index]
        return len(index)


class LHSSampler(PointSampler):
    """Will create a simple latin hypercube sampling in the given domain.
    Only works for the inner part of a domain, not the boundary!

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 

    Notes
    -----
    A bounding box is used tp create the lhs-points in the domain.
    Points outside will be rejected and additional random uniform points will be 
    added to get a total number of n_points.
    """
    def __init__(self, domain, n_points):
        assert not isinstance(domain, BoundaryDomain), \
            """LHS sampling is not implemented for boundaries."""
        super().__init__(n_points=n_points)
        self.domain = domain

    def _sample_points(self, **params):
        num_of_params = self._extract_tensor_len_from_dict(params)
        point_dict = None
        for i in range(num_of_params):
            ith_params = self._extract_points_from_dict(i, params)
            bounding_box = self.domain.bounding_box(**ith_params)
            lhs_in_box = self._create_lhs_in_bounding_box(bounding_box)
            new_points, num_valid = self._check_lhs_inside(lhs_in_box, ith_params)
            self._append_random_points(new_points, num_valid, ith_params)
            point_dict = self._set_point_dict(point_dict, new_points)
        return point_dict

    def _create_lhs_in_bounding_box(self, bounding_box):
        lhs_points = torch.zeros((self.n_points, self.domain.dim))
        # for each axis apply the lhs strategy
        for i in range(self.domain.dim):
            axis_grid = torch.linspace(bounding_box[2*i], bounding_box[2*i+1], 
                                       steps=self.n_points+1)[:-1] # dont need endpoint
            axis_length = bounding_box[2*i+1] - bounding_box[2*i]
            random_shift = axis_length/self.n_points * torch.rand(self.n_points)
            axis_points = torch.add(axis_grid, random_shift)
            # change order of points, to get 'lhs-grid' at the end
            permutation = torch.randperm(self.n_points)
            lhs_points[:, i] = axis_points[permutation]
        return lhs_points

    def _check_lhs_inside(self, lhs_points, ith_params):
        new_points = Points(lhs_points, self.domain.space)
        repeat_params = self._repeat_input_params(self.n_points, **ith_params)
        new_points.join(repeat_params)
        inside = self.domain._contains(new_points)
        index = torch.where(inside)[0]
        for key, data in new_points.items():
            new_points[key] = data[index]
        return new_points, len(index)

    def _append_random_points(self, new_points, num_valid, current_params):
        if num_valid == self.n_points:
            return 
        random_sampler = RandomUniformSampler(domain=self.domain,
                                              n_points=self.n_points-num_valid)
        random_points = random_sampler.sample_points(**current_params)
        self._append_point_dict(new_points, random_points)