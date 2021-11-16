"""Samplers for plotting and for animations of data.
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
    (including the boundary). Mostly used for plotting,
    for animations use the AnimationSampler.

    Parameters
    ----------
    plot_domain : Domain
        The domain over which the model/function should later be plotted.
        Will create points inside and at the boundary of the domain.
    n_points : int
        The number of points that should be used for the plot.
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
    def __init__(self, plot_domain, n_points, device='cpu',
                 dic_for_other_variables={}):
        assert not isinstance(plot_domain, BoundaryDomain)
        super().__init__(n_points)
        self.domain = plot_domain
        self.dic_for_other_variables = dic_for_other_variables
        self.device = device
        self._evaluate_domain()
        self.sampler = self.construct_sampler()

    def sample_points(self):
        plot_dict = self.sampler.sample_points()
        plot_dict = self._transform_to_torch_tensor(plot_dict)
        plot_dict = self._add_other_variables(plot_dict)
        return plot_dict

    def construct_sampler(self):
        if isinstance(self.domain, Interval):
            return self._construct_sampler_for_Interval(self.domain)
        inner_n_points = self._compute_inner_number_of_points()
        inner_sampler = GridSampler(self.domain, inner_n_points)
        outer_sampler = GridSampler(self.domain.boundary,
                                    len(self)-inner_n_points)
        return inner_sampler + outer_sampler

    def _construct_sampler_for_Interval(self, domain):
        left_sampler = GridSampler(domain.boundary_left, 1)
        inner_sampler = GridSampler(domain, len(self)-2)
        right_sampler = GridSampler(domain.boundary_right, 1)
        return left_sampler + inner_sampler + right_sampler       

    def _compute_inner_number_of_points(self):
        n_root = int(np.ceil(len(self)**(1/self.domain.dim)))
        n_root -= 2
        return n_root**self.domain.dim

    def _evaluate_domain(self):
        if callable(self.domain):
            self.domain = self.domain(self.dic_for_other_variables)

    def _transform_to_torch_tensor(self, plot_dict):
        for vname in plot_dict:
            plot_dict[vname] = torch.FloatTensor(plot_dict[vname],
                                                 device=self.device)
            plot_dict[vname].requires_grad = True
        return plot_dict

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
                raise ValueError(f"""Values for variables have to be a number, 
                                     list, array or tensor, but found {type(data)}.""")
            plot_dict[vname].requires_grad = True
        return plot_dict
    

class AnimationSampler(PlotSampler):
    
    def __init__(self, plot_domain, n_points, animation_domain, 
                 frame_number, device='cpu', dic_for_other_variables=None):
        self.animation_domain = animation_domain
        self.frame_number = frame_number
        super().__init__(plot_domain, n_points, device, dic_for_other_variables)
        assert isinstance(animation_domain, Interval), \
            f"""Needs a Interval as the animation_domain, found
                {type(animation_domain).__name__}"""
        self.animation_sampler = \
            self._construct_sampler_for_Interval(self.animation_domain)
        
    def _evaluate_domain(self):
        if callable(self.animation_domain):
            self.animation_domain = \
                self.animation_domain(self.dic_for_other_variables)

    def __len__(self):
        return self.n_points * self.frame_number

    def sample_points(self):
        output = [] 
        
        
        return output