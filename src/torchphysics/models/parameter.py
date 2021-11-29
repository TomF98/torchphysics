import torch

from ..problem.spaces import Points

class Parameter(Points):
    def __init__(self, init, space, **kwargs):
        init = torch.as_tensor(init).float().reshape(1, -1)
        data = torch.nn.Parameter(init)
        super().__init__(data, space, **kwargs)
