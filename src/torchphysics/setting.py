from typing import Iterable, OrderedDict
from pytorch_lightning import LightningDataModule as DataModule
import torch
import numpy as np

from .problem.problem import Problem
from .problem.spaces import Space, R1
from .problem.parameters.parameter import ParameterSub


class SimpleDataset(torch.utils.data.Dataset):
    """
    A dummy dataset which simply returns whole batches.
    Used to enable faster training with the whole dataset on a gpu.

    Parameters
    ----------
    data : A dictionary that contains batched data vor every condition
        and variables
    iterations : The number of iterations that should be performed in training.
    """
    def __init__(self, data, iterations):
        super().__init__()
        self.data = data
        self.epoch_len = iterations

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        return self.data


class Setting(Problem, DataModule):
    """
    A PDE setting, built of all variables and conditions in this problem.

    TODO: enable varying batches, i.e. non-full-batch-mode and maybe even a mode
    for datasets that are too large for gpu mem

    Parameters
    ----------
    domain : Domain
        The whole domain of this DE Problem. The Domain of the Problem is
        the product of the domains of each variable.
        E.g. C = Circle(R2('x'), ...), I = Interval(R1('t'), 0, 1), so 
        Domain = C * I
    train_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are used in training
    val_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are tracked during validation
    parameters : list or dict of Parameter objects
        Parameters can be part of conditions and still be learned together with the
        solution during training.
    solution_space : Space, optional
        The space of the desired solutions.
        E.g. if we want to solve a Navier-Stokes-eq. and find the velocity 'v' and
        pressure 'p': 
        In 2D: solution_space = R2('v') * R1('p')
        In 3D: solution_space = R3('v') * R1('p')
    n_iterations : int
        Number of iterations per epoch.
    """
    def __init__(self, domain, train_conditions={}, val_conditions={},
                 parameters={}, solution_space=R1('u'), n_iterations=1000):
        DataModule.__init__(self)
        self.n_iterations = n_iterations

        self.domain = domain
        Problem.__init__(self,
                         train_conditions=train_conditions,
                         val_conditions=val_conditions)

        self.parameters = torch.nn.ParameterDict()
        self._change_parameter_to_dict(parameters)

        # define the solution spaces
        self.solution_space = solution_space
        # run data preparation manually
        self.prepare_data()

    def _change_parameter_to_dict(self, parameters):
        if isinstance(parameters, (list, tuple)):
            for i in parameters:
                self.add_parameter(i)
        elif isinstance(parameters, dict):
            for i_name in parameters:
                self.add_parameter(parameters[i_name])
        elif isinstance(parameters, Space):
            self.add_parameter(parameters)
        else:
            raise TypeError(f"""Got type {type(parameters)} but expected
                             one of list, tuple, dict or ParameterSub.""")

    """Methods to load data with lightning"""

    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        self.train_data = {}
        train_conditions = self.get_train_conditions()
        for name in train_conditions:
            self.train_data[name] = train_conditions[name].get_data()

        self.val_data = {}
        val_conditions = self.get_val_conditions()
        for name in val_conditions:
            self.val_data[name] = val_conditions[name].get_data()

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        if stage == 'fit' or stage is None:
            self.train_data = self._setup_stage(
                self.get_train_conditions(), self.train_data)
            self.val_data = self._setup_stage(
                self.get_val_conditions(), self.val_data)
        if stage == 'validate':
            self.val_data = self._setup_stage(
                self.get_val_conditions(), self.val_data)

    def _setup_stage(self, conditions, data):
        for cn in data:
            for arg in data[cn]:
                # create torch tensor if necessary
                if isinstance(data[cn][arg], np.ndarray):
                    data[cn][arg] = torch.from_numpy(data[cn][arg])
                # enable gradient tracking if necessary
                if arg in self.variables:
                    if isinstance(conditions[cn].track_gradients, bool):
                        data[cn][arg].requires_grad = conditions[cn].track_gradients
                    elif isinstance(conditions[cn].track_gradients, Iterable):
                        data[cn][arg].requires_grad = arg in conditions[cn].track_gradients
                    else:
                        raise TypeError(
                            f'track_gradients of {arg} should be either bool or iterable.')
        return data

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.train_data, self.n_iterations),
            batch_size=None,
            num_workers=0
        )
        return dataloader

    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        dataloader = torch.utils.data.DataLoader(
            SimpleDataset(self.val_data, 1),
            batch_size=None,
            num_workers=0
        )
        return dataloader

    """Problem methods"""

    def add_parameter(self, param):
        assert isinstance(param, ParameterSub), f"{param} should be a ParameterSub obj."
        self.parameters[param.get_name()] = param

    def add_train_condition(self, condition):
        """Adds and registers a condition that is used during training
        """
        assert condition.name not in self.train_conditions, \
            f"{condition.name} cannot be present twice."
        condition.setting = self
        self.train_conditions[condition.name] = condition

    def add_val_condition(self, condition):
        """Adds and registers a condition that is used for validation
        """
        assert condition.name not in self.val_conditions, \
            f"{condition.name} cannot be present twice."
        condition.setting = self
        self.val_conditions[condition.name] = condition

    def get_train_conditions(self):
        """Returns all training conditions present in this problem.
        This does also include conditions in the variables.
        """
        return self.train_conditions

    def get_val_conditions(self):
        """Returns all validation conditions present in this problem.
        This does also include conditions in the variables.
        """
        return self.val_conditions

    def is_well_posed(self):
        raise NotImplementedError

    def get_dim(self):
        d = 0
        for v in self.domain.space:
            d += self.domain.space[v]
        return d

    @property
    def variable_dims(self):
        return self.domain.space

    @property
    def solution_dims(self):
        return self.solution_space

    @property
    def normalization_dict(self):
        out = {}
        bounds = self.domain.bounding_box()
        for var in self.domain.space:
            var_dim = self.domain.space[var_dim]
            width = []
            center = []
            for i in range(var_dim):
                min_, max_ = bounds[2*i], bounds[2*i+1]
                width.append(max_-min_)
                center.append((max_+min_)/2)
            out[var] = (width, center)
        return out

    def serialize(self):
        dct = {}
        dct['name'] = 'Setting'
        dct['n_iterations'] = self.n_iterations
        v_dict = {}
        for v_name in self.variables:
            v_dict[v_name] = self.variables[v_name].serialize()
        dct['variables'] = v_dict
        t_c_dict = {}
        for c_name in self.train_conditions:
            t_c_dict[c_name] = self.train_conditions[c_name].serialize()
        dct['train_conditions'] = t_c_dict
        v_c_dict = {}
        for c_name in self.val_conditions:
            v_c_dict[c_name] = self.val_conditions[c_name].serialize()
        dct['val_conditions'] = v_c_dict
        p_dict = {}
        for p_name in self.parameters:
            p_dict[p_name] = list(self.parameters[p_name].shape)
        dct['parameters'] = p_dict
        dct['solution_dims'] = self.solution_dims
        return dct
