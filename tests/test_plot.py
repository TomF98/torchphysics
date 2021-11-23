import pytest
import torch
import matplotlib.pyplot as pyplot

import torchphysics.utils.plot as plt  
from torchphysics.problem.domains import Interval, Circle, Parallelogram
from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.samplers import PlotSampler
from torchphysics.models.fcn import SimpleFCN


def plt_func(u):
    return torch.linalg.norm(u, dim=1)


def test_plot_create_info_text():
    input = {'t': 3, 'D': 34}
    text = plt._create_info_text(input)
    assert text == 't = 3\nD = 34'
    input = {}
    text = plt._create_info_text(input)
    assert text == ''


def test_plot_triangulation_of_domain():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0.0], [0, 1])
    ps = PlotSampler(domain, n_points=200)
    domain_points = ps.sample_points()
    triangulation = plt._triangulation_of_domain(domain,
                                                 domain_points['x'].detach().numpy()) 
    assert len(triangulation.x) == len(domain_points['x'])
    assert len(triangulation.y) == len(domain_points['x'])
    points = torch.column_stack((torch.FloatTensor(triangulation.x),
                                 torch.FloatTensor(triangulation.y)))
    assert all(torch.logical_or(domain._contains({'x':points}),
                                domain.boundary._contains({'x': points})))


def test_Plotter():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0.0], [0, 1])
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    assert plotter.plot_function == plt_func
    assert plotter.point_sampler == ps
    assert plotter.angle == [30, 30]
    assert plotter.log_interval == None
    assert plotter.plot_type == ''


def test_1D_plot():
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x'
    pyplot.close(fig)


def test_line_plot_with_wrong_function_shape():
    def wrong_shape(u):
        return u
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=wrong_shape, point_sampler=ps, 
                          plot_type='line')
    model = SimpleFCN(variable_dims={'x':1},
                      solution_dims={'u':2},
                      width=5, depth=1)
    with pytest.raises(ValueError):
        _ = plotter.plot(model=model)  


def test_1D_plot_with_textbox():
    domain = Interval(R1('x'), 0, 1)
    ps = PlotSampler(domain, n_points=200, dic_for_other_variables={'t': 2})
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':1, 't': 1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x'
    pyplot.close(fig)


def test_2D_plot():
    domain = Parallelogram(R1('x')*R1('y'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(domain, n_points=200)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':1, 'y':1},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'y'
    pyplot.close(fig)


def test_2D_plot_for_booleandomain():
    domain = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    domain -= Circle(R2('x'), [0.5, 0.5], 0.2)
    ps = PlotSampler(domain, density=0.1)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':2},
                      solution_dims={'u':1},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_scatter():
    #1D
    I = Interval(R1('x'), 0, 1)
    data = I.sample_grid(20)
    fig = plt._scatter(I.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    #2D
    I = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    data = I.sample_grid(20)
    fig = plt._scatter(I.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    #mixed
    R = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    I = Interval(R1('t'), 0, 1)
    domain = R * I
    data = domain.sample_random_uniform(20)
    fig = plt._scatter(domain.space, data)
    assert fig.axes[0].get_xlabel() == 'x'
    assert fig.axes[0].get_ylabel() == 'x'
    assert fig.axes[0].get_zlabel() == 't'
    pyplot.close(fig)


def test_2D_quiver():
    def quiver_plt(u):
        return u.detach().cpu().numpy()
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, density=0.1)
    plotter = plt.Plotter(plot_function=quiver_plt, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':2},
                      solution_dims={'u':2},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_2D_quiver_with_textbox():
    def quiver_plt(u):
        return u.detach().cpu().numpy()
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, density=0.1, dic_for_other_variables={'t': 2})
    plotter = plt.Plotter(plot_function=quiver_plt, point_sampler=ps)
    model = SimpleFCN(variable_dims={'x':2, 't':1},
                      solution_dims={'u':2},
                      width=5, depth=1)
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'x_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'x_2'
    pyplot.close(fig)


def test_3D_curve():
    I = Interval(R1('i'), -1, 2)
    ps = PlotSampler(I, density=0.1, dic_for_other_variables={'t': 2})
    plotter = plt.Plotter(plot_function=lambda u:u, point_sampler=ps)
    model = SimpleFCN(variable_dims={'i': 1, 't':1},
                      solution_dims={'u': 2},
                      width=5, depth=1)  
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-1.15, 2.15)
    assert fig.axes[0].get_xlabel() == 'i'
    pyplot.close(fig)


def test_contour_2D():
    P = Parallelogram(R2('R'), [0, 0], [1, 0], [0, 2])
    model = SimpleFCN(variable_dims={'R': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)    
    ps = PlotSampler(P, n_points=500)
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps,
                          plot_type='contour_surface')
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_contour_2D_with_textbox():
    P = Parallelogram(R2('R'), [0, 0], [1, 0], [0, 2])
    model = SimpleFCN(variable_dims={'R': 2, 't': 2},
                      solution_dims={'u': 2},
                      width=5, depth=1)    
    ps = PlotSampler(P, n_points=500, dic_for_other_variables={'t': [2.0, 0.0]})
    plotter = plt.Plotter(plot_function=plt_func, point_sampler=ps,
                          plot_type='contour_surface')
    fig = plotter.plot(model=model)  
    assert fig.axes[0].get_xlim() == (-0.05, 1.05)
    assert fig.axes[0].get_xlabel() == 'R_1'
    assert fig.axes[0].get_ylim() == (-0.1, 2.1)
    assert fig.axes[0].get_ylabel() == 'R_2'
    pyplot.close(fig)


def test_contour_plot_with_wrong_function_shape():
    def wrong_shape(u):
        return u
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 2])
    ps = PlotSampler(P, n_points=200)
    plotter = plt.Plotter(plot_function=wrong_shape, point_sampler=ps, 
                          plot_type='contour_surface')
    model = SimpleFCN(variable_dims={'x':2},
                      solution_dims={'u':2},
                      width=5, depth=1)
    with pytest.raises(ValueError):
        _ = plotter.plot(model=model)  