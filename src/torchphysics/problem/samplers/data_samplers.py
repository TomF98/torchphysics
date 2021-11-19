"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""
import torch
import numpy as np

from .sampler_base import PointSampler


class DataSampler(PointSampler):
    """A sampler that processes external created data points.

    Parameters
    ----------
    input_data : dictionary
        A dictionary containing the input data for the model. The dictionary keys
        have to fit the space variables of the underlying problem. 
        If the data is present in an array/tensor, but ordered accordingly, 
        one can use the methode space.embed(...) or domain.space.embed(...) 
        to create the dictionary.
    output_data : list, tensor or array
        The expected model values at the given input data.   
    """

    def __init__(self, input_data, output_data):
        self._check_input_is_dict(input_data)
        self.input_data = input_data
        self.output_data = self._data_to_tensor(output_data)
        n = self._extract_tensor_len_from_dict(input_data)
        super().__init__(n_points=n)

    def _check_input_is_dict(self, input_data):
        if not isinstance(input_data, dict):
            raise TypeError(f"""The input_data has to be dictionary,
                                but found {type(input_data)}. If the data is a
                                list/array/tensor one can use the 'space.embed'
                                methode to create the dictionary.""")
    
    def _data_dict_to_tensor_dict(self, data_dict):
        tensor_dict = {}
        for key, data in data_dict.items():
            tensor_dict[key] = self._data_to_tensor(data)
        return tensor_dict

    def _data_to_tensor(self, init):
        if isinstance(init, torch.Tensor):
            data = init
        elif isinstance(init, np.ndarray):
            data = torch.from_numpy(init)
        elif isinstance(init, (tuple, list)):
            data = torch.Tensor(init)
        elif isinstance(init, float):
            data = torch.Tensor((init,))
        elif isinstance(init, int):
            data = torch.Tensor((float(init),))
        return data

    def sample_points(self, **params):
        return {**self.input_data, 'target': self.output_data}