"""Samplers for plotting and animations of model outputs.
"""
import numpy as np
import torch
import numbers

from ..domains.domain import BoundaryDomain
from ..domains import Interval
from .sampler_base import PointSampler
from .grid_samplers import GridSampler


class PlotSampler(PointSampler):
    """A sampler that creates a equidistant point grid over a domain
    (including the boundary). Only used for plotting,

    Parameters
    ----------
    plot_domain : Domain
        The domain over which the model/function should later be plotted.
        Will create points inside and at the boundary of the domain.
    n_points : int, optional
        The number of points that should be used for the plot.
    density : float, optional
        The desiered density of the created points.
    device : str or torch device, optional
        The device of the model/function.
    dic_for_other_variables : dict, optional
        Since the plot will only evaluate the model at a specific point, 
        the values for all other variables are needed. 
        E.g. {'t' : 1, 'D' : [1,2], ...}

    Notes
    -----
    Can also be used to create your own PlotSampler. By either changing the
    used sampler after the initialization (self.sampler=...) or by creating 
    your own class that inherits from PlotSampler.
    """
    def __init__(self, plot_domain, n_points=None, density=None, device='cpu',
                 dic_for_other_variables={}):
        assert not isinstance(plot_domain, BoundaryDomain), \
            "Plotting for boundarys not implemented"""
        super().__init__(n_points=n_points, density=density)
        self.dic_for_other_variables = dic_for_other_variables
        self.device = device
        self._evaluate_domain(plot_domain)
        self.sampler = self.construct_sampler()

    def _evaluate_domain(self, plot_domain):
        tensor_variable_dict = self._transform_input_dict_to_tensor_dict()
        self.domain = plot_domain(**tensor_variable_dict)

    def _transform_input_dict_to_tensor_dict(self):
        tranformed_dict = {}
        for vname, value in self.dic_for_other_variables.items():
            if not isinstance(value, torch.Tensor):
                tranformed_dict[vname] = torch.tensor(value)
            else:
                tranformed_dict[vname] = value
        return tranformed_dict

    def construct_sampler(self):
        if self.n_points:
            return self._plot_sampler_with_n_points()
        else: # density is used
            return self._plot_sampler_with_density()

    def _plot_sampler_with_n_points(self):
        if isinstance(self.domain, Interval):
            return self._construct_sampler_for_Interval(self.domain, n=self.n_points)
        inner_n_points = self._compute_inner_number_of_points()
        inner_sampler = GridSampler(self.domain, inner_n_points)
        outer_sampler = GridSampler(self.domain.boundary, len(self)-inner_n_points)
        return inner_sampler + outer_sampler

    def _plot_sampler_with_density(self):
        if isinstance(self.domain, Interval):
            return self._construct_sampler_for_Interval(self.domain, d=self.density)
        inner_sampler = GridSampler(self.domain, density=self.density)
        outer_sampler = GridSampler(self.domain.boundary, density=self.density)
        return inner_sampler + outer_sampler

    def _construct_sampler_for_Interval(self, domain, n=None, d=None):
        left_sampler = GridSampler(domain.boundary_left, 1)
        inner_sampler = GridSampler(domain, n_points=n, density=d)
        right_sampler = GridSampler(domain.boundary_right, 1)
        return left_sampler + inner_sampler + right_sampler       

    def _compute_inner_number_of_points(self):
        n_root = int(np.ceil(len(self)**(1/self.domain.dim)))
        n_root -= 2
        return n_root**self.domain.dim

    def sample_points(self):
        plot_points = self.sampler.sample_points()
        num_of_points = self._extract_tensor_len_from_dict(plot_points)
        self.set_length(num_of_points)
        self._set_device_and_grad_true(plot_points)
        self._add_other_variables(plot_points)
        return plot_points

    def _set_device_and_grad_true(self, plot_dict):
        for vname in plot_dict:
            plot_dict[vname].requires_grad = True
            plot_dict[vname].to(self.device)

    def _add_other_variables(self, plot_dict):
        for vname in self.dic_for_other_variables:
            data = self.dic_for_other_variables[vname]
            if isinstance(data, numbers.Number):
                plot_dict[vname] = float(data) * torch.ones((len(self), 1),
                                                            device=self.device)
            elif isinstance(data, (list, np.ndarray, torch.Tensor)):
                data_length = len(data)
                array = data * np.ones((len(self), data_length))
                plot_dict[vname] = torch.FloatTensor(array, device=self.device) 
            else:
                raise TypeError(f"""Values for variables have to be a number, 
                                     list, array or tensor, but found {type(data)}.""")
            plot_dict[vname].requires_grad = True
        return plot_dict