"""File with some explicit samplers that can always be used.
"""
import numpy as np
import numbers

from .sampler_base import DataSampler
from ..domains.domain import BoundaryDomain
from ..domains.domain1D import Interval


class GridSampler(DataSampler):
    """Will sample equdistant grid-points in the given domain.

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    """
    def __init__(self, domain, n_points):
        super().__init__(n_points)
        self.domain = domain

    def sample_points(self):
        return self.domain.sample_grid(len(self))


class SpacedGridSampler(DataSampler):
    """Will sample non equdistant grid points in the given interval.
    Onyl works on intervals!

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
        super().__init__(n_points)
        self.domain = domain
        self.exponent = exponent

    def sample_points(self):
        points = np.linspace(0, 1, len(self)+2)[1:-1]
        if self.exponent > 1:
            points = points**self.exponent
        else:
            points = 1 - points**(1/self.exponent)
        length = self.domain.upper_bound - self.domain.lower_bound
        points = points * length + self.domain.lower_bound
        return  self.domain.divide_points_to_space_variables(points.reshape(-1, 1))  


class RandomUniformSampler(DataSampler):
    """Will sample random uniform distributed points in the given domain.

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    """
    def __init__(self, domain, n_points):
        super().__init__(n_points)
        self.domain = domain

    def sample_points(self):
        return self.domain.sample_random_uniform(len(self))


class GaussianSampler(DataSampler):
    """Will sample normal/gaussian distributed points in the given domain.

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    mean : list or array
        The center/mean of the distribution. Has to fit the dimension
        of the given domain.
    cov : number, list or array
        The (co-)variance of the distribution. For dimensions >= 2,
        cov should be a symmetric and positive-semidefinite matrix.
        If a number is given as an input, the covariance will be set
        to cov*unit-matrix.
    """
    def __init__(self, domain, n_points, mean, cov):
        super().__init__(n_points)
        self.domain = domain
        self.mean = mean
        self.cov = cov
        self._check_mean_correct_dim()

    def _check_mean_correct_dim(self):
        if isinstance(self.mean, numbers.Number):
            self.mean = [self.mean]
        if isinstance(self.domain, BoundaryDomain):
            assert len(self.mean) == self.domain.dim + 1, \
                   f"""Mean {self.mean} does not fit the domain"""
        else:
            assert len(self.mean) == self.domain.dim, \
                   f"""Mean {self.mean} does not fit the domain"""

    def sample_points(self):
        if isinstance(self.domain, BoundaryDomain):
            return self._sample_on_boundary()
        return self._sample_inside()

    def _sample_inside(self):
        if isinstance(self.cov, numbers.Number):
            self.cov = self.cov * np.eye(self.domain.dim)
        points = np.zeros((len(self), self.domain.dim))
        current_ind = 0
        while current_ind < len(self):
            new_points = np.random.multivariate_normal(self.mean,
                                                       self.cov,
                                                       size=len(self)-current_ind)
            current_ind = self._update_points(points, current_ind, new_points)
        return self.domain.divide_points_to_space_variables(points)

    def _update_points(self, points, current_ind, new_points):
        inside = self.domain.is_inside(new_points)
        index = np.where(inside)[0]
        new_ind = current_ind + len(index)
        points[current_ind:new_ind] = new_points[index]
        return new_ind

    def _sample_on_boundary(self):
        if self.domain.dim == 1: # the boundary in 2D is only one dimensional
            return self._sample_2D_boundary()
        raise NotImplementedError

    def _sample_2D_boundary(self):
        # first find the position at the boundary
        outline = self.domain.outline()
        posi_on_bound, vertices, min_len, max_len = \
            self._compute_position_on_2D_bound(outline)
        assert not posi_on_bound == -1, f"""Mean {self.mean} not on boundary"""
        # now compute the normal distribution in a interval and 'wrap' it 
        # around the domain-boundary
        line_points = np.random.normal(posi_on_bound, self.cov, size=len(self))
        self._scale_to_boundary_part(line_points, min_len, max_len)
        line_points = np.sort(line_points)
        points = np.zeros((len(self), 2))
        points, _, _ = self.domain._distribute_line_to_boundary(points, index=0,
                                                                line_points=line_points,
                                                                corners=vertices,
                                                                current_length=min_len)
        return self.domain.divide_points_to_space_variables(points)
    
    def _compute_position_on_2D_bound(self, outline):
        # Walk on each boundary part and check if the 
        # point lays on this section.
        found = False
        min_len, max_len = 0, 0
        for boundary_part in outline:
            min_len += max_len - min_len
            posi_on_bound, side_length, found = \
                self._check_boundary_part(found, min_len, boundary_part) 
            max_len += sum(side_length)
            if found:
                break
        return posi_on_bound, boundary_part, min_len, max_len

    def _check_boundary_part(self, found, min_len, boundary_part):
        posi_on_bound = -1
        side_length = np.zeros(len(boundary_part))
        dist_mean_vertex = np.linalg.norm(self.mean - boundary_part[0])
        for i in range(len(boundary_part)-1):
            side_length[i] = np.linalg.norm(boundary_part[i+1] - boundary_part[i])
            if not found:
                dist_mean_next_vertex = np.linalg.norm(self.mean - boundary_part[i+1])
                if self._check_dist(dist_mean_vertex, dist_mean_next_vertex,
                                    side_length[i]):
                    posi_on_bound = min_len + sum(side_length[:i]) + dist_mean_vertex
                    found = True
                dist_mean_vertex = dist_mean_next_vertex
        return posi_on_bound, side_length, found

    def _check_dist(self, dist_mean_vertex, dist_mean_next_vertex, side_length):
        distance = dist_mean_vertex + dist_mean_next_vertex - side_length
        return distance <= 3*self.domain.tol                

    def _scale_to_boundary_part(self, line_points, min_len, max_len):
        # after the normal distribution some points could be smaller
        # or bigger then the perimeter of the boundary section.
        # Therefore scale with the perimeter.
        perimeter = max_len - min_len
        smaller_min = np.where(line_points < min_len)[0]
        while not len(smaller_min) == 0:
            line_points[smaller_min] += perimeter
            smaller_min = np.where(line_points < 0)[0]
        bigger_perim = np.where(line_points > max_len)[0]
        while not len(bigger_perim) == 0:
            line_points[bigger_perim] -= perimeter
            bigger_perim = np.where(line_points > max_len)[0]


class LHSSampler(DataSampler):
    """Will create a simple latin hypercube sampling in the given domain.

    Parameters
    ----------
    domain : Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 

    Notes
    -----
    If points should be created inside a domain, we use a bounding box and create
    the lhs-points in the box. Points outside will be rejected and additional 
    random uniform points will be added to get a total number of n_points.
    """
    def __init__(self, domain, n_points):
        super().__init__(n_points)
        self.domain = domain

    def sample_points(self):
        if isinstance(self.domain, BoundaryDomain):
            return self._sample_lhs_on_boundary()
        return self._sample_lhs_inside()

    def _sample_lhs_inside(self):
        bounds = self.domain.bounding_box()
        points = self._sample_lhs_in_cube(bounds=bounds)
        inside = self.domain.is_inside(points)
        index = np.where(inside)[0]
        points = self.domain._check_grid_enough_points(len(self), points[index])
        return self.domain.divide_points_to_space_variables(points)     

    def _sample_lhs_in_cube(self, bounds, permutate=True):
        lhs_axis = np.zeros((len(self), self.domain.dim))
        # divide each axis and compute lhs
        for i in range(self.domain.dim):
            divide_axis = np.linspace(bounds[2*i], bounds[2*i + 1],
                                      len(self), endpoint=False)
            axis_len = bounds[2*i + 1] - bounds[2*i]
            random_points = np.random.uniform(0, axis_len/len(self), len(self))
            new_axis = np.add(divide_axis, random_points)
            if permutate:
                lhs_axis[:, i] = np.random.permutation(new_axis)
            else:
                lhs_axis[:, i] = new_axis
        return lhs_axis.astype(np.float32)

    def _sample_lhs_on_boundary(self):
        if self.domain.dim == 1: # the boundary in 2D is only one dimensional
            return self._sample_lhs_on_2D_boundary()
        raise NotImplementedError

    def _sample_lhs_on_2D_boundary(self):
        bounds = [0, self.domain.domain.polygon.boundary.length]
        line_points = self._sample_lhs_in_cube(bounds=bounds, permutate=False)
        return self.domain._transform_points_to_boundary(len(self), line_points)