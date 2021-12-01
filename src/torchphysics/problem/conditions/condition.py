"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc

import torch
import numpy as np

from ...models import Parameter
from ...utils import UserFunction

"""
TODO: check creation of tensors on correct devices, maybe additional flags
    are necessary
"""

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


class DataCondition(Condition):
    """
    A condition that fits a single given module to data (handed through a PyTorch
    dataloader).

    Parameters
    -------
    module : torch.nn.Module
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
        return self.norm(self.module(x) - y)


class PINNCondition(Condition):
    """
    A condition that minimizes the residual of a single module in a differential equation,
    like it is common in the concept of Physics-Informed Neural Networks (PINNs) [1].
    
    Parameters
    -------
    module : torch.nn.Module
        The torch module which should solve the differential equation.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the differential equation.
    residual_fn : callable
        A user-defined function that computes the residual from inputs and outputs
        of the model, e.g. by using utils.differentialoperators
    norm : torch.nn.Module
        A torch Module which computes the (scalar) norm of the given residual,
        e.g. torch.nn.MSELoss().
    parameter : Parameter
        A Parameter that can be used in the residual_fn and should be learned in
        parallel, e.g. based on data.
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
    def __init__(self, module, sampler, residual_fn, norm=2, parameter=Parameter.empty(),
                 name='pinncondition', weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=True)
        self.module = module
        self.parameter = parameter
        self.sampler = sampler
        self.residual_fn = UserFunction(residual_fn)
        self.norm = norm
    
    def forward(self):
        x = next(self.sampler)
        y = self.module(x)  # y is in coords of output space
        return torch.mean(self.residual_fn({**y.coordinates,
                                            **x.coordinates,
                                            **self.parameter.coordinates})**self.norm,
                          dim=0)**(1/self.norm)


class DeepRitzCondition(Condition):
    """
    A Condition that minimizes the variational problem of a PDE (weak formulation),
    similar to the idea presented in [1].

    def poisson_residual(u, x):
        return 0.5*(torch.sum(grad(u)**2), dim=1)

    Parameters
    ----------
    module : torch.nn.Module
        The torch module which should solve the differential equation.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the differential equation.
    integrand : callable
        A user-defined function that computes the integrand of the weak formulation,
        e.g. by using utils.differentialoperators
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
    ..  [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017
    """
    def __init__(self, module, sampler, residual_fn, parameter=Parameter.empty(),
                 name='deepritzcondition', weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=True)
        self.module = module
        self.parameter = parameter
        self.sampler = sampler
        self.residual_fn = UserFunction(residual_fn)
    
    def forward(self):
        x = next(self.sampler)
        y = self.module(x)
        return torch.mean(self.residual_fn({**y.coordinates,
                                            **x.coordinates,
                                            **self.parameter.coordinates}),
                          dim=0)


"""
Old conditions:
"""

# """ """ class RestrictionCondition(Condition):
#     """A condition to limit the solution/parameter values to specific range.
#     E.g. solution > 0 or Parameter D in [2, 6].
    
#     Parameters
#     ----------
#     restriction_fun: function handle
#         A method that takes the output and input of a model and returns the desiered 
#         restriction. If the restriction is fullfilled the function has to return 0.
#         E.g. the Parameter D should be bigger 1:
#         |   restriction_fun(D):
#         |       return torch.min(torch.tensor([0, D-1]))
#     norm : torch.nn.Module
#         A Pytorch module which forward pass returns the scalar norm of the difference of
#         two input tensors, and is therefore similar to the implementation of nn.MSELoss.
#         The norm is used to compute the loss of this restriction.
#     point_sampler: DataSampler, optional
#         A sampler that creates the training/validation points for this condition.
#     name : str
#         name of this condition (should be unique per condition)
#     weight : float, optional
#         Scalar weight of this condition that is used in the weighted sum for the
#         training loss. Defaults to 1.
#     boundary_domain : Domain
#         If the restriction is connected to the normal vector of a domain, the 
#         boundary has to be given as an input. (So the normals can be computed) 
#         The input for the normal vector in the restriction function will be
#         'normal'.
#     track_gradients : bool, optional
#         If True, the gradients are still tracked during validation to enable the
#         computation of derivatives w.r.t. the inputs.
#     data_plot_variables : bool or tuple, optional
#         The variables which are used to log the used training data in a scatter plot.
#         If False, no plots are created. If True, behaviour is defined in each condition.
#     """

#     def __init__(self, restriction_fun, norm, point_sampler=None,
#                  name='restriction', weight=1.0, boundary_domain=None,
#                  track_gradients=False, data_plot_variables=False):
#         super().__init__(name, norm, weight, point_sampler=point_sampler, 
#                          track_gradients=track_gradients,
#                          data_plot_variables=data_plot_variables)
#         self.restriction_fun = restriction_fun
#         self.boundary_domain = boundary_domain

#     def forward(self, model, data):
#         data_dic = {**self.setting.parameters}
#         if self.point_sampler is not None:
#             u = model({v: data[v] for v in self.setting.domain.space})
#             data_dic = {**u, **data, **self.setting.parameters}
#         inp = prepare_user_fun_input(self.restriction_fun, data_dic)
#         err = self.restriction_fun(**inp)
#         return self.norm(err, torch.zeros_like(err))

#     def get_data(self):
#         out = {}
#         if self.point_sampler is not None:
#             inp_data = self.point_sampler.sample_points()
#             out = {**inp_data}
#             if self.boundary_domain is not None:
#                 normals = self.boundary_domain.normals(inp_data)
#                 out['normal'] = normals
#         return out

#     def get_data_plot_variables(self):
#         if self.data_plot_variables is True:
#             return self.setting.domain.space
#         elif self.data_plot_variables is False:
#             return None
#         else:
#             return self.data_plot_variables


# class BoundaryCondition(Condition):
#     """
#     Parent class for all boundary conditions.

#     Parameters
#     ----------
#     name : str
#         name of this condition (should be unique per condition)
#     norm : torch.nn.Module
#         A Pytorch module which forward pass returns the scalar norm of the difference of
#         two input tensors, and is therefore similar to the implementation of nn.MSELoss.
#         The norm is used to compute the loss for the deviation of the model from the
#         given data.
#     point_sampler: DataSampler
#         A sampler that creates the training/validation points for this condition.
#     track_gradients : bool
#         If True, the gradients are still tracked during validation to enable the
#         computation of derivatives w.r.t. the inputs.
#     weight : float, optional
#         Scalar weight of this condition that is used in the weighted sum for the
#         training loss. Defaults to 1.
#     data_plot_variables : bool or tuple, optional
#         The variables which are used to log the used training data in a scatter plot.
#         If False, no plots are created. If True, behaviour is defined in each condition.
#     """

#     def __init__(self, name, norm, point_sampler, boundary_domain, 
#                  track_gradients, weight=1.0, data_plot_variables=True):
#         super().__init__(name, norm, weight=weight,
#                          point_sampler=point_sampler,
#                          track_gradients=track_gradients,
#                          data_plot_variables=data_plot_variables)
#         self.boundary_domain = boundary_domain

#     def get_data_plot_variables(self):
#         if self.data_plot_variables is True:
#             return self.boundary_domain
#         elif self.data_plot_variables is False:
#             return None
#         else:
#             return self.data_plot_variables

#     def serialize(self):
#         dct = super().serialize()
#         dct['boundary_domain'] = self.boundary_domain
#         return dct


# class DirichletCondition(BoundaryCondition):
#     """
#     Implementation of a Dirichlet boundary condition based on a function handle.

#     Parameters
#     ----------
#     dirichlet_fun : function handle
#         A method that takes boundary points (in the usual dictionary form) as an input
#         and returns the desired boundary values at those points.
#     name : str
#         name of this condition (should be unique per condition)
#     norm : torch.nn.Module
#         A Pytorch module which forward pass returns the scalar norm of the difference of
#         two input tensors, and is therefore similar to the implementation of nn.MSELoss.
#         The norm is used to compute the loss for the deviation of the model from the
#         given data.
#     point_sampler: DataSampler
#         A sampler that creates the training/validation points for this condition.
#     solution_name : str, optional
#         The output function for which the given boundary condition should be learned.
#     data_fun_whole_batch : bool, optional
#         Specifies if the dirichlet_fun can work with a whole batch of data (then True),
#         or every sample point has to be evaluated alone (False). 
#     weight : float, optional
#         Scalar weight of this condition that is used in the weighted sum for the
#         training loss. Defaults to 1.
#     data_plot_variables : bool or tuple, optional
#         The variables which are used to log the used training data in a scatter plot.
#         If False, no plots are created. If True, behaviour is defined in each condition.
#     """

#     def __init__(self, dirichlet_fun, name, norm, point_sampler,
#                  solution_name='u', whole_batch=True,
#                  weight=1.0, data_plot_variables=True):
#         super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
#                          track_gradients=False, boundary_domain=None,
#                          data_plot_variables=data_plot_variables)
#         self.solution_name = solution_name
#         self.dirichlet_fun = dirichlet_fun
#         self.whole_batch = whole_batch

#     def forward(self, model, data):
#         u = model({v: data[v] for v in self.setting.domain.space})
#         return self.norm(u[self.solution_name], data['target'])

#     def get_data(self):
#         inp_data = self.point_sampler.sample_points()
#         inp_data, target = apply_data_fun(self.dirichlet_fun,
#                                           inp_data,
#                                           whole_batch=self.whole_batch,
#                                           batch_size=len(self.point_sampler))
#         return {**inp_data,
#                 'target': target}

#     def serialize(self):
#         dct = super().serialize()
#         dct['dirichlet_fun'] = self.dirichlet_fun.__name__
#         return dct


# class NeumannCondition(BoundaryCondition):
#     """
#     Implementation of a Neumann boundary condition based on a function handle.

#     Parameters
#     ----------
#     neumann_fun : function handle
#         A method that takes boundary points (in the usual dictionary form) as an input
#         and returns the desired values of the normal derivatives of the model.
#     name : str
#         name of this condition (should be unique per condition)
#     norm : torch.nn.Module
#         A Pytorch module which forward pass returns the scalar norm of the difference of
#         two input tensors, and is therefore similar to the implementation of nn.MSELoss.
#         The norm is used to compute the loss for the deviation of the model from the
#         given data.
#     point_sampler: DataSampler
#         A sampler that creates the training/validation points for this condition.
#     solution_name : str, optional
#         The output function for which the given boundary condition should be learned.
#     data_fun_whole_batch : bool, optional
#         Specifies if the neumann_fun can work with a whole batch of data (then True),
#         or every sample point has to be evaluated alone (False). 
#     weight : float, optional
#         Scalar weight of this condition that is used in the weighted sum for the
#         training loss. Defaults to 1.
#     data_plot_variables : bool or tuple, optional
#         The variables which are used to log the used training data in a scatter plot.
#         If False, no plots are created. If True, behaviour is defined in each condition.
#     """

#     def __init__(self, neumann_fun, name, norm, point_sampler, boundary_domain,
#                  solution_name='u', whole_batch=True,
#                  weight=1.0, data_plot_variables=True):
#         super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
#                          track_gradients=False, boundary_domain=boundary_domain,
#                          data_plot_variables=data_plot_variables)
#         self.solution_name = solution_name
#         self.neumann_fun = neumann_fun
#         self.whole_batch = whole_batch

#     def forward(self, model, data):
#         u = model({v: data[v] for v in self.setting.domain.space})
#         normal_derivatives = normal_derivative(u[self.solution_name],
#                                                data[self.boundary_domain],
#                                                data['normal'])
#         return self.norm(normal_derivatives, data['target'])

#     def get_data(self):
#         inp_data = self.point_sampler.sample_points()
#         normals = self.boundary_domain.normals(inp_data)
#         inp_data, target = apply_data_fun(self.neumann_fun,
#                                           inp_data,
#                                           whole_batch=self.whole_batch,
#                                           batch_size=len(self.point_sampler))
#         return {**inp_data,
#                 'target': target,
#                 'normal': normals}

#     def serialize(self):
#         dct = super().serialize()
#         dct['neumann_fun'] = self.neumann_fun.__name__
#         return dct


# class DiffEqBoundaryCondition(BoundaryCondition):
#     """
#     Implementation a arbitrary boundary condition based on a function handle.

#     Parameters
#     ----------
#     bound_condition_fun : function handle
#         A method that takes the output and input (in the usual dictionary form)
#         of a model, the boundary normals and additional data (given through
#         data_fun, and
#         only when needed) as an input. The method then computes and returns 
#         the desired boundary condition.
#     name : str
#         name of this condition (should be unique per condition)
#     norm : torch.nn.Module
#         A Pytorch module which forward pass returns the scalar norm of the difference of
#         two input tensors, and is therefore similar to the implementation of nn.MSELoss.
#         The norm is used to compute the loss for the deviation of the model from the
#         given data.
#     point_sampler: DataSampler
#         A sampler that creates the training/validation points for this condition.
#     weight : float, optional
#         Scalar weight of this condition that is used in the weighted sum for the
#         training loss. Defaults to 1.
#     data_fun : function handle, optional
#         A method that represents the right-hand side of the boundary condition. As
#         an input it takes the boundary points in the usual dictionary form.
#         If the right-hand side is independent of the model, it is more efficient to
#         compute the values only once and save them.
#         If the right-hand side dependents on the model outputs, or is zero, this
#         parameter should be None and the whole condition has to be implemented in
#         bound_condition_fun.
#     data_fun_whole_batch : bool, optional
#         Specifies if the data_fun can work with a whole batch of data (then True)
#         or every sample point has to be evaluated alone (False). 
#     data_plot_variables : bool or tuple, optional
#         The variables which are used to log the used training data in a scatter plot.
#         If False, no plots are created. If True, behaviour is defined in each condition.
#     """

#     def __init__(self, bound_condition_fun, name, norm, point_sampler,
#                  boundary_domain, data_fun=None, weight=1.0, solution_name='u',
#                  data_fun_whole_batch=True, data_plot_variables=True):
#         super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
#                          track_gradients=False, boundary_domain=boundary_domain,
#                          data_plot_variables=data_plot_variables)
#         self.bound_condition_fun = bound_condition_fun
#         self.data_fun = data_fun
#         self.data_fun_whole_batch = data_fun_whole_batch

#     def forward(self, model, data):
#         u = model({v: data[v] for v in self.setting.domain.space})
#         inp = prepare_user_fun_input(self.bound_condition_fun,
#                                      {**u,
#                                       **data,
#                                       **self.setting.parameters})
#         err = self.bound_condition_fun(**inp)
#         return self.norm(err, torch.zeros_like(err))

#     def get_data(self):
#         inp_data = self.point_sampler.sample_points()
#         normals = self.boundary_domain.normals(inp_data)
#         if self.data_fun is None:
#             return {**inp_data,
#                     'normal': normals}
#         else:
#             inp_data, data = apply_data_fun(self.data_fun,
#                                             {**inp_data, 'normal': normals},
#                                             whole_batch=self.data_fun_whole_batch,
#                                             batch_size=len(self.point_sampler))
#             return {**inp_data,
#                     'data': data,
#                     'normal': normals}

#     def serialize(self):
#         dct = super().serialize()
#         dct['bound_condition_fun'] = self.bound_condition_fun.__name__
#         if self.data_fun is not None:
#             dct['data_fun'] = self.data_fun.__name__
#         return dct


# def apply_data_fun(f, args, whole_batch=True, batch_size=None):
#     # typical steps to apply a user-defined data function:
#     # 1) filter the input arguments required by the given function
#     inp = prepare_user_fun_input(f, args)
#     # 2) if the function is defined entry-wise, we wrap it in a for loop
#     if whole_batch:
#         out = f(**inp)
#     else:
#         out = apply_to_batch(f, batch_size=batch_size, **inp)
#     # 3) data points that evaluated to None or NaN should not be used
#     if whole_batch:
#         batch_size = len(out)
#     args, out = remove_nan(args, out, batch_size)
#     return args, out


# def remove_nan(inp, out, batch_size):
#     # remove input and output data where the operation evaluated to NaN
#     keep = ~(np.isnan(out).any(axis=tuple(range(1, len(np.shape(out))))))
#     if np.any(~keep):
#         print(f"""Warning: {np.sum(~keep)} values will be removed from the data because
#                   the given data_fun evaluated to None or NaN. Please make sure this is
#                   the desired behaviour.""")
#     for v in inp:
#         if is_batch(inp[v], batch_size):
#             inp[v] = inp[v][keep]
#     out = out[keep]
#     return inp, out


# def get_data_len(size):
#     if isinstance(size, int):
#         return size
#     elif isinstance(size, (tuple, list)):
#         return np.prod(size)
#     elif isinstance(size, dict):
#         return np.prod(list(size.values()))
#     else:
#         raise ValueError(f"""'dataset_size should be one of int,
#                              tuple, list or dict. Got {type(size)}.""") """ """