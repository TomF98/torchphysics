"""The basic structure of every sampler and all sampler 'operations'.
"""
import abc
import torch

from ...utils.user_fun import UserFunction


class PointSampler:
    """Handles the creation and interconnection of training/validation points.

    Parameters
    ----------
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
    def __init__(self, n_points=None, density=None, filter=None):
        self.n_points = n_points
        self.density = density
        self.length = None
        if filter:
            self.filter = UserFunction(filter)
        else:
            self.filter = None 

    def set_length(self, length):
        """If a density is used, the number of points will not be known before
        hand. If len(PointSampler) is needed one can set the expected number 
        of points here.

        Parameters
        ----------
        length : int
            The expected number of points that this sampler will create.

        Notes
        -----
        If the domain is independent of other variables and a density is used, the 
        sampler will, after the first call to 'sampler_points', set this value itself. 
        """
        self.length = length

    def __len__(self):
        if self.n_points:
            return self.n_points
        elif self.length:
            return self.length
        else:
            raise ValueError("""The expected number of samples is not known yet. 
                                Set the length by using .set_length, if this 
                                property is needed""")

    def sample_points(self, **params):
        if self.filter:
            return self._sample_points_with_filter(**params)
        else:
            return self._sample_points(**params)

    @abc.abstractmethod
    def _sample_points_with_filter(self, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_points(self, **params):
        raise NotImplementedError

    def __mul__(self, other):
        assert isinstance(other, PointSampler)
        # returns a sampler that samples from the 'cartesian product'
        # of the samples of two samplers
        return ProductSampler(self, other)

    def __add__(self, other):
        assert isinstance(other, PointSampler)
        # returns a sampler that samples from two samplers
        return ConcatSampler(self, other)

    def append(self, other):
        assert isinstance(other, PointSampler)
        return AppendSampler(self, other)

    def _repeat_input_params(self, n, **params):
        repeated_params = params
        if n > 1:
            for key, domain_param in params.items():
                # will repeat like ([[a], [b]]) -> ([[a], [a], [a], [b], [b], [b]])
                repeated_params[key] = torch.repeat_interleave(domain_param, n, dim=0)
        return repeated_params

    def _repeat_sampled_points(self, n, point_dict):
        repeated_params = point_dict
        if n > 1:
            for key, points in point_dict.items():
                # will repeat like ([[a], [b]]) -> ([[a], [b]], [[a], [b]], [[a], [b]])
                repeated_params[key] = points.repeat(n, 1)
        return repeated_params

    def _extract_tensor_len_from_dict(self, point_dict):
        tensor_len = 1
        for key in point_dict:
            tensor_len = len(point_dict[key])
            break
        return tensor_len

    def _sample_params_independent(self, sample_function, **params):
        """If the domain is independent of the used params it is more efficent
        to sample points once and then copy them accordingly.
        """
        points = sample_function(n=self.n_points, d=self.density)
        num_of_points = self._extract_tensor_len_from_dict(points)
        self.set_length(num_of_points)
        num_of_params = self._extract_tensor_len_from_dict(params)
        repeated_params = self._repeat_input_params(num_of_points, **params)
        grid_points = self._repeat_sampled_points(num_of_params, points)
        return {**grid_points, **repeated_params}

    def _sample_params_dependent(self, sample_function, **params):
        """If the domain is dependent on some params, we can't always sample points
        for all params at once. Therefore we need a loop to iterate over the params.
        This happens for example with denstiy sampling or grid sampling. 
        """
        num_of_params = self._extract_tensor_len_from_dict(params)
        sample_dict = None
        for i in range(num_of_params):
            ith_params = self._extract_points_from_dict(i, params)
            new_points = sample_function(self.n_points, self.density, **ith_params)
            num_of_points = self._extract_tensor_len_from_dict(new_points)
            repeated_params = self._repeat_input_params(num_of_points, **ith_params)
            if not sample_dict:
                sample_dict = {**new_points, **repeated_params}
            else:
                self._append_point_dict(sample_dict, {**new_points, **repeated_params})
        return sample_dict

    def _extract_points_from_dict(self, i, params):
        ith_params = {}
        for key in params.keys():
            ith_params[key] = params[key][i]
        return ith_params

    def _append_point_dict(self, sample_dict, new_point_dic):
        for key in sample_dict.keys():
            sample_dict[key] = torch.cat((sample_dict[key], new_point_dic[key]), dim=0)


class ProductSampler(PointSampler):
    """A sampler that constructs the product of two samplers.
    Will create a meshgrid of the data points of both samplers.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected.
    """
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        return len(self.sampler_a) * len(self.sampler_b)

    def sample_points(self, **params):
        b_points = self.sampler_b.sample_points(**params)
        a_points = self.sampler_a.sample_points(**params, **b_points)
        return a_points


class ConcatSampler(PointSampler):
    """A sampler that adds two single samplers together.
    Will concatenate the data points of both samplers.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected.
    """
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        return len(self.sampler_a) + len(self.sampler_b)

    def sample_points(self, **params):
        samples_a = self.sampler_a.sample_points(**params)
        samples_b = self.sampler_b.sample_points(**params)
        for vname in samples_a:
            samples_a[vname] = torch.cat((samples_a[vname], samples_b[vname]), dim=0)
        return samples_a


class AppendSampler(PointSampler):
    """A sampler that appends the output of two samplers behind each other.

    Parameters
    ----------
    sampler_a, sampler_b : PointSampler
        The two PointSamplers that should be connected. Both Samplers should create 
        the same number of points.
    """
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__()

    def __len__(self):
        return len(self.sampler_a)

    def sample_points(self, **params):
        samples_a = self.sampler_a.sample_points(**params)
        samples_b = self.sampler_b.sample_points(**params)
        return  {**samples_a, **samples_b}