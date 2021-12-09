"""Function to show an example of the created points of the sampler. 
"""
import numpy as np

import matplotlib.pyplot as plt


def scatter(subspace, *samplers):
    """Shows (one batch) of used points in the training. If the sampler is
    static, the shown points will be the points for the training. If not
    the points may vary, depending of the sampler. 

    Parameters
    ----------
    subspace : torchphysics.problem.Space
        The (sub-)space of which the points should be plotted.
        Only plotting for dimensions <= 3 is possible.
    *samplers : torchphysics.problem.Samplers
        The diffrent samplers for which the points should be plotted.
        The plot for each sampler will be created in the order there were
        passed in.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        The figure handle of the plot.
    """
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