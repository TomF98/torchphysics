"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""

from .sampler_base import PointSampler
from ..spaces import Points


class DataSampler(PointSampler):
    """A sampler that processes external created data points.

    Parameters
    ----------
    input_data : torchphysics.spaces.Points or dict
        A points object containing the input data for the model.
    output_data : torchphysics.spaces.Points or dict
        The expected model values at the given input data in the
        correct output space.
    """

    def __init__(self, input_data, output_data):
        if isinstance(input_data, Points):
            self.input_data = input_data
        elif isinstance(input_data, dict):
            self.input_data = Points.from_coordinates(input_data)
        else:
            raise TypeError("input_data should be one of Points or dict.")
        if isinstance(output_data, Points):
            self.output_data = output_data
        elif isinstance(output_data, dict):
            self.output_data = Points.from_coordinates(output_data)
        else:
            raise TypeError("output_data should be one of Points or dict.")
        n = len(input_data)
        assert len(output_data) == n
        super().__init__(n_points=n)

    def sample_points(self, params=Points.empty()):
        return self.input_data, self.output_data