import torch
import torch.nn as nn

from ..problem.spaces import Points, Space

class Model(nn.Module):
    """Neural networks that can be trained to fulfill user-defined conditions.

    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    """
    def __init__(self, input_space, output_space):
        super().__init__()
        self.input_space = input_space
        self.output_space = output_space


class NormalizationLayer(Model):
    """
    A first layer that scales a domain to the range (-1, 1)^domain.dim, since this
    can improve convergence during training.

    Parameters
    ----------
    domain : Domain
        The domain from which this layer expects sampled points. The layer will use
        its bounding box to compute the normalization factors.
    """
    def __init__(self, domain, requires_grad=True):
        super().__init__(input_space=domain.space, output_space=domain.space)
        self.normalize = nn.Linear(domain.space.dim, domain.space.dim)

        box = domain.bounding_box
        mins = box[::2]
        maxs = box[1::2]

        # compute width and center
        diag = []
        bias = []
        for i in range(domain.dim):
            diag.append(maxs[i] - mins[i])
            bias.append((maxs[i] + mins[i])/2)
        
        diag = 2./torch.tensor(diag)
        bias = -torch.tensor(bias)*diag
        with torch.no_grad():
            self.normalize.weight.copy_(torch.diag(diag))
            self.normalize.bias.copy_(bias)

    def forward(self, points):
        return Points(self.normalize(points), self.output_space)

class AdaptiveWeightLayer(Model):
    """
    Adds adaptive weights to the non-reduced loss. The weights are maximized by
    reversing the gradients, similar to the idea in [1].
    Should currently only be used with fixed points.

    Parameters
    ----------
    space : Space
        The output space in which the loss lies
    n : int
        The amount of sampled points in each batch.

    Notes
    -----
    ..  [1] L. McClenny, "Self-Adaptive Physics-Informed Neural Networks using a Soft
        Attention Mechanism", 2020.
    """
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    @classmethod
    def grad_reverse(cls, x):
        return cls.GradReverse.apply(x)

    def __init__(self, space, n):
        super().__init__(space, space)
        self.weight = torch.nn.Parameter(
            torch.ones(n, space.dim)
        )
    
    def forward(self, points):
        weight = self.grad_reverse(self.weight)
        return Points(weight*points, self.output_space)



class Parallel(Model):
    """
    A model that wraps multiple models which should be applied in parallel.
    """
    def __init__(self, *models):
        input_space = Space({})
        output_space = Space({})
        for model in models:
            assert input_space.keys().isdisjoint(model.input_space)
            assert output_space.keys().isdisjoint(model.output_space)
            input_space = input_space * model.input_space
            output_space = output_space * model.output_space
        super().__init__(input_space, output_space)
        self.models = nn.ModuleList(models)
    
    def forward(self, points):
        out = []
        for model in self.models:
            out.append(model(points[:, list(model.input_space.keys())]).as_tensor)
        return Points(torch.cat((out), dim=0), self.output_space)

class Sequential(Model):
    """
    A model that wraps multiple models which should be applied sequentially.
    """
    def __init__(self, *models):
        super().__init__(models[0].input_space, models[-1].output_space)
        self.sequential = nn.Sequential(*models)
    
    def forward(self, points):
        return self.sequential(points)
