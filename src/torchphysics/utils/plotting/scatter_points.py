"""Function to show an example of the created points of the sampler. 
"""
import numpy as np

import matplotlib.pyplot as plt


def scatter(subspace, *samplers):

    assert subspace.dim <= 3, "Can only scatter points in dimensions <= 3."

    fig, ax, scatter_fn = _choose_scatter_function(subspace.dim)
    ax.grid()
    for sampler in samplers:
        points = sampler.sample_points()[:, list(subspace.keys())]
        numpy_points = points.as_tensor.detach().cpu().numpy()
        scatter_fn(ax, numpy_points)
    return fig


def _choose_scatter_function(space_dim):
    fig = plt.figure()
    if space_dim == 1:
        ax = fig.add_subplot()
        return fig, ax, _scatter_1D
    elif space_dim == 2:
        ax = fig.add_subplot()
        return fig, ax, _scatter_2D    
    else:
        ax = fig.add_subplot(projection='3d')
        return fig, ax, _scatter_3D  


def _scatter_1D(ax, points):
    ax.scatter(points, np.zeros_like(points))


def _scatter_2D(ax, points):
    ax.scatter(points[:, 0], points[:, 1])


def _scatter_3D(ax, points):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])