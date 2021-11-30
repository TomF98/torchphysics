"""The basic structure of every sampler and all sampler 'operations'.
"""
import abc
import torch
import warnings

from ...utils.user_fun import UserFunction
from ..spaces.points import Points


class PointSampler:
    """Handles the creation and interconnection of training/validation points.

    Parameters
    ----------
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desiered density of the created points.
    filter_fn : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return false. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """
    def __init__(self, n_points=None, density=None, filter_fn=None):
        self.n_points = n_points
        self.density = density
        self.length = None
        if filter_fn:
            self.filter_fn = UserFunction(filter_fn)
        else:
            self.filter_fn = None 

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
        sampler will, after the first call to 'sample_points', set this value itself. 
        """
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample_points()

    def __len__(self):
        if self.length:
            return self.length
        elif self.n_points:
            return self.n_points
        else:
            raise ValueError("""The expected number of samples is not known yet. 
                                Set the length by using .set_length, if this 
                                property is needed""")

    def sample_points(self, params=Points.empty()):
        """The methode that creates the points.

        Parameters
        ----------
        params : torchphysics.spaces.Points
            Additional parameters for the domain.

        Returns
        -------
        Points:
            A Points-Object containing the created points and, if parameters were 
            passed as an input, the parameters. Whereby the input parameters
            will get repeated, so that each row of the tensor corresponds to  
            valid point in the given (product) domain.
        """
        if self.filter_fn:
            return self._sample_points_with_filter(params)
        else:
            return self._sample_points(params)

    @abc.abstractmethod
    def _sample_points_with_filter(self, params=Points.empty()):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_points(self, params=Points.empty()):
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

    def _sample_params_independent(self, sample_function, params):
        """If the domain is independent of the used params it is more efficent
        to sample points once and then copy them accordingly.
        """
        points = sample_function(n=self.n_points, d=self.density)
        num_of_points = len(points)
        self.set_length(num_of_points)
        num_of_params = max(1, len(params))
        repeated_params = self._repeat_params(params, num_of_points)
        repeated_points = points.repeat(num_of_params, 1)
        return repeated_points.join(repeated_params)

    def _sample_params_dependent(self, sample_function, params):
        """If the domain is dependent on some params, we can't always sample points
        for all params at once. Therefore we need a loop to iterate over the params.
        This happens for example with denstiy sampling or grid sampling. 
        """
        num_of_params = max(1, len(params))
        sample_points = None
        for i in range(num_of_params):
            new_points = self._sample_for_ith_param(sample_function, params, i)
            sample_points = self._set_sampled_points(sample_points, new_points)
        return sample_points

    def _sample_for_ith_param(self, sample_function, params, i):
        ith_params = params[i, ]
        new_points = sample_function(self.n_points, self.density, ith_params)
        num_of_points = len(new_points)
        repeated_params = self._repeat_params(ith_params, num_of_points)
        return new_points.join(repeated_params)

    def _set_sampled_points(self, sample_points, new_points):
        if not sample_points:
            return new_points
        return sample_points | new_points

    def _repeat_params(self, params, n):
        repeated_params = Points(torch.repeat_interleave(params, n, dim=0),
                                 params.space)
        return repeated_params

    def _apply_filter(self, sample_points):
        filter_true = self.filter_fn(sample_points)
        index = torch.where(filter_true)[0]
        return sample_points[index, ]

    def _check_iteration_number(self, iterations, num_of_new_points):
        if iterations == 10:
            warnings.warn(f"""Sampling points with filter did run 10
                              iterations and until now only found 
                              {num_of_new_points} from {self.n_points} points.
                              This may take some time.""")  
        elif iterations >= 20 and num_of_new_points == 0:
            raise RuntimeError("""Run 20 iterations and could not find a single 
                                  valid point for the filter condition.""") 

    def _cut_tensor_to_length_n(self, points):
        return points[:self.n_points, ]


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
        if self.length:
            return self.length
        return len(self.sampler_a) * len(self.sampler_b)

    def sample_points(self, params=Points.empty()):
        b_points = self.sampler_b.sample_points(params)
        a_points = self.sampler_a.sample_points(b_points.join(params))
        self.set_length(len(a_points))
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
        if self.length:
            return self.length
        return len(self.sampler_a) + len(self.sampler_b)

    def sample_points(self, params=Points.empty()):
        samples_a = self.sampler_a.sample_points(params)
        samples_b = self.sampler_b.sample_points(params)
        self.set_length(len(samples_a) + len(samples_b))
        return samples_a | samples_b


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
        if self.length:
            return self.length
        return len(self.sampler_a)

    def sample_points(self, params=Points.empty()):
        samples_a = self.sampler_a.sample_points(params)
        samples_b = self.sampler_b.sample_points(params)
        self.set_length(len(samples_a))
        return samples_a.join(samples_b)