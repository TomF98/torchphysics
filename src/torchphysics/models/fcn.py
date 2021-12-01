from typing import Iterable
import torch
import torch.nn as nn

from .model import Model
from ..problem.spaces import Points


class FCN(Model):
    """
    A simple fully connected neural network.

    """
    def __init__(self,
                 input_space,
                 output_space,
                 hidden=(20,20,20),
                 activations=nn.Tanh(),
                 xavier_gains=5/3):
        super().__init__(input_space, output_space)

        if not isinstance(activations, (list, tuple)):
            activations = len(hidden) * [activations]
        if not isinstance(xavier_gains, (list, tuple)):
            xavier_gains = len(hidden) * [xavier_gains]

        layers = []
        layers.append(nn.Linear(self.input_space.dim, hidden[0]))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[0])
        layers.append(activations[0])
        for i in range(len(hidden)-1):
            layers.append(nn.Linear(hidden[i], hidden[i+1]))
            torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[i+1])
            layers.append(activations[i+1])
        layers.append(nn.Linear(hidden[-1], self.output_space.dim))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=1)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        return Points(self.sequential(points), self.output_space)
