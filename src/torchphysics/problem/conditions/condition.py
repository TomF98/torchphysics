"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc

import torch
import numpy as np

from ...utils import (normal_derivative,
                     prepare_user_fun_input,
                     apply_to_batch,
                     is_batch)


class Condition(torch.nn.Module):
    """
    A Condition that should be fulfilled by the DE solution.

    Conditions can be applied to the boundary or inner part of the DE domain and are a
    central concept in this library. Their solution can either be enforced during
    training or tracked during validation.

    Parameters
    ----------
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used for the computation of the conditioning loss/metric.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    track_gradients : bool or list of str or list of DiffVariables
        Whether the gradients w.r.t. the inputs should be tracked.
        Tracking can be necessary for training of a PDE.
        If True, all gradients will be tracked.
        If a list of strings or variables is passed, gradient is tracked
        only for the variables in the list.
        If False, no gradients will be tracked.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, name, norm, point_sampler, weight=1.0,
                 track_gradients=True,
                 data_plot_variables=True):
        super().__init__()
        self.name = name
        self.norm = norm
        self.weight = weight
        self.point_sampler = point_sampler
        self.track_gradients = track_gradients
        self.data_plot_variables = data_plot_variables

        # variables are registered when the condition is added to a setting
        self.setting = None

    @abc.abstractmethod
    def get_data(self):
        """Creates and returns the data for the given condition."""
        return

    @abc.abstractmethod
    def get_data_plot_variables(self):
        return

    def serialize(self):
        dct = {}
        dct['name'] = self.name
        dct['norm'] = self.norm.__class__.__name__
        dct['weight'] = self.weight
        return dct


class DiffEqCondition(Condition):
    """
    A condition that enforces the solution of a Differential Equation in the
    inner part of a domain.

    Parameters
    ----------
    pde : function handle
        A method that takes the output and input of a model and computes its deviation
        from some (partial) differential equation. See utils.differentialoperators for
        useful helper functions.
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from a PDE.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    name : str
        name of this condition (should be unique per condition)
    data_fun : function handle, optional
        A method that represents the possible right-hand side of the differential 
        equation. Since it is more efficent to comput the function only
        once, if it is independetn of the model. 
        If the right-hand side dependents on the model outputs, or is zero, this
        parameter should be None and the whole condition has to be implemented in
        the pde!
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    track_gradients : bool, optional
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs.
    data_fun_whole_batch : bool, optional
        Specifies if the data_fun can work with a whole batch of data (then True)
        or every sample point has to be evaluated alone (False). 
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, pde, norm, point_sampler, 
                 name='pde', data_fun=None, weight=1.0,
                 track_gradients=True, data_fun_whole_batch=True,
                 data_plot_variables=False):
        super().__init__(name, norm, weight, point_sampler=point_sampler, 
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        self.pde = pde
        self.data_fun = data_fun
        self.data_fun_whole_batch = data_fun_whole_batch

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.domain.space})
        inp = prepare_user_fun_input(self.pde,
                                     {**u,
                                      **data,
                                      **self.setting.parameters})
        err = self.pde(**inp)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        inp_data = self.point_sampler.sample_points()
        if self.data_fun is None:
            return inp_data
        else:
            inp_data, data = apply_data_fun(self.data_fun,
                                            inp_data,
                                            whole_batch=self.data_fun_whole_batch,
                                            batch_size=len(self.point_sampler))
            return {**inp_data,
                    'data': data}

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.setting.domain.space
        elif self.data_plot_variables is False:
            return None
        else:
            return self.data_plot_variables
        

class DataCondition(Condition):
    """
    A condition that enforces the model to fit a given dataset.

    Parameters
    ----------
    data_inp : dict
        A dictionary containing pairs of variables and data for that variables,
        organized in numpy arrays or torch tensors of equal length.
    data_out : array-like
        The targeted solution values for the data points in data_inp.
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    solution_name : str, optional
        The output function which should be fitted to the given dataset.
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    """

    def __init__(self, data_inp, data_out, name, norm,
                 solution_name='u', weight=1.0):
        super().__init__(name, norm, weight, point_sampler=None,
                         track_gradients=False,
                         data_plot_variables=False)
        self.solution_name = solution_name
        self.data_x = data_inp
        self.data_u = data_out

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.domain.space})
        return self.norm(u[self.solution_name], data['target'])

    def get_data(self):
        return {**self.data_x,
                'target': self.data_u}

    def get_data_plot_variables(self):
        return None


class RestrictionCondition(Condition):
    """A condition to limit the solution/parameter values to specific range.
    E.g. solution > 0 or Parameter D in [2, 6].
    
    Parameters
    ----------
    restriction_fun: function handle
        A method that takes the output and input of a model and returns the desiered 
        restriction. If the restriction is fullfilled the function has to return 0.
        E.g. the Parameter D should be bigger 1:
        |   restriction_fun(D):
        |       return torch.min(torch.tensor([0, D-1]))
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss of this restriction.
    point_sampler: DataSampler, optional
        A sampler that creates the training/validation points for this condition.
    name : str
        name of this condition (should be unique per condition)
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    boundary_domain : Domain
        If the restriction is connected to the normal vector of a domain, the 
        boundary has to be given as an input. (So the normals can be computed) 
        The input for the normal vector in the restriction function will be
        'normal'.
    track_gradients : bool, optional
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs.
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, restriction_fun, norm, point_sampler=None,
                 name='restriction', weight=1.0, boundary_domain=None,
                 track_gradients=False, data_plot_variables=False):
        super().__init__(name, norm, weight, point_sampler=point_sampler, 
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        self.restriction_fun = restriction_fun
        self.boundary_domain = boundary_domain

    def forward(self, model, data):
        data_dic = {**self.setting.parameters}
        if self.point_sampler is not None:
            u = model({v: data[v] for v in self.setting.domain.space})
            data_dic = {**u, **data, **self.setting.parameters}
        inp = prepare_user_fun_input(self.restriction_fun, data_dic)
        err = self.restriction_fun(**inp)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        out = {}
        if self.point_sampler is not None:
            inp_data = self.point_sampler.sample_points()
            out = {**inp_data}
            if self.boundary_domain is not None:
                normals = self.boundary_domain.normals(inp_data)
                out['normal'] = normals
        return out

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.setting.domain.space
        elif self.data_plot_variables is False:
            return None
        else:
            return self.data_plot_variables


class BoundaryCondition(Condition):
    """
    Parent class for all boundary conditions.

    Parameters
    ----------
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    track_gradients : bool
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs.
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, name, norm, point_sampler, boundary_domain, 
                 track_gradients, weight=1.0, data_plot_variables=True):
        super().__init__(name, norm, weight=weight,
                         point_sampler=point_sampler,
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        self.boundary_domain = boundary_domain

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.boundary_domain
        elif self.data_plot_variables is False:
            return None
        else:
            return self.data_plot_variables

    def serialize(self):
        dct = super().serialize()
        dct['boundary_domain'] = self.boundary_domain
        return dct


class DirichletCondition(BoundaryCondition):
    """
    Implementation of a Dirichlet boundary condition based on a function handle.

    Parameters
    ----------
    dirichlet_fun : function handle
        A method that takes boundary points (in the usual dictionary form) as an input
        and returns the desired boundary values at those points.
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    solution_name : str, optional
        The output function for which the given boundary condition should be learned.
    data_fun_whole_batch : bool, optional
        Specifies if the dirichlet_fun can work with a whole batch of data (then True),
        or every sample point has to be evaluated alone (False). 
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, dirichlet_fun, name, norm, point_sampler,
                 solution_name='u', whole_batch=True,
                 weight=1.0, data_plot_variables=True):
        super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
                         track_gradients=False, boundary_domain=None,
                         data_plot_variables=data_plot_variables)
        self.solution_name = solution_name
        self.dirichlet_fun = dirichlet_fun
        self.whole_batch = whole_batch

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.domain.space})
        return self.norm(u[self.solution_name], data['target'])

    def get_data(self):
        inp_data = self.point_sampler.sample_points()
        inp_data, target = apply_data_fun(self.dirichlet_fun,
                                          inp_data,
                                          whole_batch=self.whole_batch,
                                          batch_size=len(self.point_sampler))
        return {**inp_data,
                'target': target}

    def serialize(self):
        dct = super().serialize()
        dct['dirichlet_fun'] = self.dirichlet_fun.__name__
        return dct


class NeumannCondition(BoundaryCondition):
    """
    Implementation of a Neumann boundary condition based on a function handle.

    Parameters
    ----------
    neumann_fun : function handle
        A method that takes boundary points (in the usual dictionary form) as an input
        and returns the desired values of the normal derivatives of the model.
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    solution_name : str, optional
        The output function for which the given boundary condition should be learned.
    data_fun_whole_batch : bool, optional
        Specifies if the neumann_fun can work with a whole batch of data (then True),
        or every sample point has to be evaluated alone (False). 
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, neumann_fun, name, norm, point_sampler, boundary_domain,
                 solution_name='u', whole_batch=True,
                 weight=1.0, data_plot_variables=True):
        super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
                         track_gradients=False, boundary_domain=boundary_domain,
                         data_plot_variables=data_plot_variables)
        self.solution_name = solution_name
        self.neumann_fun = neumann_fun
        self.whole_batch = whole_batch

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.domain.space})
        normal_derivatives = normal_derivative(u[self.solution_name],
                                               data[self.boundary_domain],
                                               data['normal'])
        return self.norm(normal_derivatives, data['target'])

    def get_data(self):
        inp_data = self.point_sampler.sample_points()
        normals = self.boundary_domain.normals(inp_data)
        inp_data, target = apply_data_fun(self.neumann_fun,
                                          inp_data,
                                          whole_batch=self.whole_batch,
                                          batch_size=len(self.point_sampler))
        return {**inp_data,
                'target': target,
                'normal': normals}

    def serialize(self):
        dct = super().serialize()
        dct['neumann_fun'] = self.neumann_fun.__name__
        return dct


class DiffEqBoundaryCondition(BoundaryCondition):
    """
    Implementation a arbitrary boundary condition based on a function handle.

    Parameters
    ----------
    bound_condition_fun : function handle
        A method that takes the output and input (in the usual dictionary form)
        of a model, the boundary normals and additional data (given through
        data_fun, and
        only when needed) as an input. The method then computes and returns 
        the desired boundary condition.
    name : str
        name of this condition (should be unique per condition)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    point_sampler: DataSampler
        A sampler that creates the training/validation points for this condition.
    weight : float, optional
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    data_fun : function handle, optional
        A method that represents the right-hand side of the boundary condition. As
        an input it takes the boundary points in the usual dictionary form.
        If the right-hand side is independent of the model, it is more efficient to
        compute the values only once and save them.
        If the right-hand side dependents on the model outputs, or is zero, this
        parameter should be None and the whole condition has to be implemented in
        bound_condition_fun.
    data_fun_whole_batch : bool, optional
        Specifies if the data_fun can work with a whole batch of data (then True)
        or every sample point has to be evaluated alone (False). 
    data_plot_variables : bool or tuple, optional
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, bound_condition_fun, name, norm, point_sampler,
                 boundary_domain, data_fun=None, weight=1.0, solution_name='u',
                 data_fun_whole_batch=True, data_plot_variables=True):
        super().__init__(name, norm, weight=weight, point_sampler=point_sampler,
                         track_gradients=False, boundary_domain=boundary_domain,
                         data_plot_variables=data_plot_variables)
        self.bound_condition_fun = bound_condition_fun
        self.data_fun = data_fun
        self.data_fun_whole_batch = data_fun_whole_batch

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.domain.space})
        inp = prepare_user_fun_input(self.bound_condition_fun,
                                     {**u,
                                      **data,
                                      **self.setting.parameters})
        err = self.bound_condition_fun(**inp)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        inp_data = self.point_sampler.sample_points()
        normals = self.boundary_domain.normals(inp_data)
        if self.data_fun is None:
            return {**inp_data,
                    'normal': normals}
        else:
            inp_data, data = apply_data_fun(self.data_fun,
                                            {**inp_data, 'normal': normals},
                                            whole_batch=self.data_fun_whole_batch,
                                            batch_size=len(self.point_sampler))
            return {**inp_data,
                    'data': data,
                    'normal': normals}

    def serialize(self):
        dct = super().serialize()
        dct['bound_condition_fun'] = self.bound_condition_fun.__name__
        if self.data_fun is not None:
            dct['data_fun'] = self.data_fun.__name__
        return dct


def apply_data_fun(f, args, whole_batch=True, batch_size=None):
    # typical steps to apply a user-defined data function:
    # 1) filter the input arguments required by the given function
    inp = prepare_user_fun_input(f, args)
    # 2) if the function is defined entry-wise, we wrap it in a for loop
    if whole_batch:
        out = f(**inp)
    else:
        out = apply_to_batch(f, batch_size=batch_size, **inp)
    # 3) data points that evaluated to None or NaN should not be used
    if whole_batch:
        batch_size = len(out)
    args, out = remove_nan(args, out, batch_size)
    return args, out


def remove_nan(inp, out, batch_size):
    # remove input and output data where the operation evaluated to NaN
    keep = ~(np.isnan(out).any(axis=tuple(range(1, len(np.shape(out))))))
    if np.any(~keep):
        print(f"""Warning: {np.sum(~keep)} values will be removed from the data because
                  the given data_fun evaluated to None or NaN. Please make sure this is
                  the desired behaviour.""")
    for v in inp:
        if is_batch(inp[v], batch_size):
            inp[v] = inp[v][keep]
    out = out[keep]
    return inp, out


def get_data_len(size):
    if isinstance(size, int):
        return size
    elif isinstance(size, (tuple, list)):
        return np.prod(size)
    elif isinstance(size, dict):
        return np.prod(list(size.values()))
    else:
        raise ValueError(f"""'dataset_size should be one of int,
                             tuple, list or dict. Got {type(size)}.""")