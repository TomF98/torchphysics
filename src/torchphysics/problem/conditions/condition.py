"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc

import torch
import numpy as np

from ...models import Parameter
from ...utils import UserFunction
from ..spaces import Points
from ..samplers import StaticSampler

class Condition(torch.nn.Module):
    """
    A general condition which should be optimized or tracked.

    Parameters
    -------
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    track_gradients : bool
        Whether to track input gradients or not. Helps to avoid tracking the
        gradients during validation. If a condition is applied during training,
        the gradients will always be tracked.
    """

    def __init__(self, name=None, weight=1.0, track_gradients=True):
        super().__init__()
        self.name = name
        self.weight = weight
        self.track_gradients = track_gradients
    
    @abc.abstractmethod
    def forward(self):
        """
        The forward run performed by this condition.

        Returns
        -------
        torch.Tensor : the loss which should be minimized or monitored during training
        """
        raise NotImplementedError

    def _track_gradients(self, points):
        points_coordinates = points.coordinates
        for var in points_coordinates:
            points_coordinates[var].requires_grad = True
        return points_coordinates, Points.from_coordinates(points_coordinates)
    
    def _setup_data_functions(self, data_functions, sampler):
        for fun in data_functions:
            data_functions[fun] = UserFunction(data_functions[fun])
        if isinstance(sampler, StaticSampler):
            # functions can be evaluated once
            for fun in data_functions:
                points = next(sampler)
                data_functions[fun] = UserFunction(data_functions[fun](points))
        return data_functions
        


class DataCondition(Condition):
    """
    A condition that fits a single given module to data (handed through a PyTorch
    dataloader).

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be fitted to data.
    dataloader : torch.utils.DataLoader
        A PyTorch dataloader which supplies the iterator to load data-target pairs
        from some given dataset. Data and target should be handed as points in input
        or output spaces, i.e. with the correct point object.
    norm : torch.nn.Module
        A torch Module which computes the (scalar) distance of the computed output
        and the given target, e.g. torch.nn.MSELoss().
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    """
    def __init__(self, module, dataloader, norm, name='datacondition', weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=False)
        self.module = module
        self.iterator = iter(dataloader)
        self.norm = norm

    def forward(self):
        x, y = next(self.iterator)
        return self.norm(self.module(x).as_tensor, y)


class ResidualCondition(Condition):
    """
    A condition that minimizes the Lp-norm of the residual of a single module, like it
    is common in the concept of Physics-Informed Neural Networks (PINNs) [1] or for
    boundary conditions in several approaches.
    
    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
        could be an inner or a boundary domain.
    residual_fn : callable
        A user-defined function that computes the residual from inputs and outputs
        of the model, e.g. by using utils.differentialoperators or domain.normal
    norm : int
        The p of the used Lp-norm
    data_functions : dict
        A dictionary of user-defined functions and their names (as keys). Can be
        used e.g. for right sides in PDEs or functions in boundary conditions.
    track_gradients : bool
        Whether gradients w.r.t. the inputs should be tracked during training or
        not. Defaults to true, since this is needed to compute differential operators
        in PINNs.
    parameter : Parameter
        A Parameter that can be used in the residual_fn and should be learned in
        parallel, e.g. based on data (in an additional DataCondition).
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.

    Notes
    -----
    ..  [1] M. Raissi, "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial differential
        equations", Journal of Computational Physics, vol. 378, pp. 686-707, 2019.
    """
    def __init__(self, module, sampler, residual_fn, norm=2, track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), name='pinncondition',
                 weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.module = module
        self.parameter = parameter
        self.sampler = sampler
        self.residual_fn = UserFunction(residual_fn)
        self.norm = norm
        self.data_functions = self._setup_data_functions(data_functions, sampler)
    
    def forward(self):
        x = next(self.sampler)
        x_coordinates, x = self._track_gradients(x)

        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)

        y = self.module(x)  # y is in coords of output space
        return torch.mean(torch.abs(self.residual_fn({**y.coordinates,
                                                      **x_coordinates,
                                                      **self.parameter.coordinates,
                                                      **data}))**self.norm,
                          dim=0)**(1/self.norm)


class PINNCondition(ResidualCondition):
    """
    Alias for :class:`ResidualCondition`.
    """

class IntegralCondition(Condition):
    """
    A Condition that minimizes the integral of a user-defined integrand, could be used
    in general variational approaches, e.g. the Deep-Ritz-Method [1].

    Parameters
    ----------
    module : torchphysics.Model
        The torch module which should solve the differential equation.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the differential equation.
    integrand : callable
        A user-defined function that computes the integrand of the weak formulation,
        e.g. by using utils.differentialoperators
    track_gradients : bool
        Whether gradients w.r.t. the inputs should be tracked during training or
        not. Defaults to true, since this is needed to compute differential operators
        in Deep-Ritz-Method.
    data_functions : dict
        A dictionary of user-defined functions and their names (as keys). Can be
        used e.g. for right sides in PDEs.
    parameter : Parameter
        A Parameter that can be used in the integrand and should be leraned in parallel,
        e.g. based on data.
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.

    Notes
    -----
    ..  The implementation differs from the PINNCondition with p=1 only in avoiding the
        computation of absolute values in the integral. The main difference of PINN and
        DeepRitz is the definition of the integrand by the user.
    ..  [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017
    
    Examples
    --------
    def poisson_residual(u, x):
        return 0.5*(torch.sum(grad(u)**2), dim=1)
    """
    def __init__(self, module, sampler, integrand_fn, track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), name='deepritzcondition',
                 weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.module = module
        self.parameter = parameter
        self.sampler = sampler
        self.integrand_fn = UserFunction(integrand_fn)
        self.data_functions = self._setup_data_functions(data_functions, sampler)
    
    def forward(self):
        x = next(self.sampler)
        x_coordinates, x = self._track_gradients(x)

        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)

        y = self.module(x)
        return torch.mean(self.integrand_fn({**y.coordinates,
                                            **x_coordinates,
                                            **self.parameter.coordinates,
                                            **data}),
                          dim=0)


class DeepRitzCondition(IntegralCondition):
    """
    Alias for :class:`IntegralCondition`.
    """


class ParameterCondition(Condition):
    """
    A condition that applies a penalty term on some parameters which are
    optimized during the training process.

    Parameters
    ----------
    parameter : torchphysics.Parameter
        The parameter that should be optimized.
    penalty : callable
        A user-defined function that defines a penalty term on the parameters.
    weight : float
        The weight multiplied with the loss of the penalty during training.
    name : str
        The name of this condition which will be monitored in logging.
    """
    def __init__(self, parameter, penalty, weight, name='parametercondition'):
        super().__init__(name=name, weight=weight, track_gradients=False)
        self.parameter = parameter
        self.penalty = UserFunction(penalty)
    
    def forward(self):
        return self.penalty(self.parameter.coordinates)
