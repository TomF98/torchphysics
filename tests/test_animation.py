import pytest
import torch
import numpy as np
import os
import matplotlib.pyplot as pyplot

import torchphysics.utils.plotting.animation as ani
from torchphysics.problem.domains import Interval, Circle, Parallelogram
from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.samplers import AnimationSampler, GridSampler
from torchphysics.models.fcn import SimpleFCN


def ani_func(u):
    return u


def test_create_animation_sampler():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.frame_number == 20
    assert ps.n_points == 20


def test_create_animation_sampler_error_with_wrong_animation_domain():
    C = Circle(R2('x'), [0, 0], 2)
    with pytest.raises(AssertionError):
        _ = AnimationSampler(plot_domain=C, animation_domain=C,
                             frame_number=20, n_points=20)


def test_animation_sampler_get_animation_key():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.animation_key == 't'


def test_animation_sampler_check_variable_dependencie():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.plot_domain_constant
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert not ps.plot_domain_constant


def test_animation_sampler_check_variable_dependencie():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert ps.plot_domain_constant
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    assert not ps.plot_domain_constant


def test_animation_sampler_create_animation_points():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    ani_points = ps.sample_animation_points()
    assert isinstance(ani_points, dict)
    assert len(ani_points['t']) == ps.frame_number


def test_animation_sampler_create_plot_domain_points_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    domain_points = ps.sample_plot_domain_points(None)
    assert isinstance(domain_points, dict)
    assert len(domain_points['x']) == len(ps)
    for data in domain_points.values():
        assert data.requires_grad 


def test_animation_sampler_create_plot_domain_points_independent_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=0.2)
    domain_points = ps.sample_plot_domain_points(None)
    assert isinstance(domain_points, dict)
    assert len(domain_points['x']) == len(ps)
    for data in domain_points.values():
        assert data.requires_grad 


def test_animation_sampler_create_plot_domain_points_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    ani_points = ps.sample_animation_points()
    domain_points = ps.sample_plot_domain_points(ani_points)
    assert isinstance(domain_points, list)
    assert len(domain_points) == ps.frame_number
    for i in range(ps.frame_number):
        for data in domain_points[i].values():
            assert data.requires_grad 


def test_create_animation_data_independent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, dict)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, dict)
    assert len(domain_points['x']) >= ps.n_points
    assert out_shape == 1


def test_create_animation_data_independent_with_additional_variable():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20,
                          dic_for_other_variables={'D': [0.0, 2.2]})
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, dict)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, dict)
    assert len(domain_points['x']) >= ps.n_points
    assert out_shape == 1


def test_create_animation_data_dependent():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, n_points=20,
                          dic_for_other_variables={'D': [0.0, 2.2]})
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, dict)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, list)
    assert len(domain_points[0]['x']) == ps.n_points
    assert out_shape == 1


def test_create_animation_data_dependent_and_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=0.3,
                          dic_for_other_variables={'D': [0.0, 2.2]})
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    animation_points, domain_points, outputs, out_shape = \
        ani._create_animation_data(model, ani_func, ps)
    assert isinstance(animation_points, dict)
    assert isinstance(outputs, list)
    assert len(outputs) == ps.frame_number
    assert isinstance(domain_points, list)
    assert out_shape == 1


def test_evaluate_animation_function():
    model = SimpleFCN(variable_dims={'x': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)   
    inp_dict = {'x': torch.tensor([[0.0, 0.2]])}
    output = ani._evaluate_animation_function(model, ani_func, inp_dict)
    assert isinstance(output, np.ndarray)


def test_evaluate_animation_function_if_no_tensor_is_used():
    model = SimpleFCN(variable_dims={'x': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)   
    inp_dict = {'x': torch.tensor([[0.0, 0.2]])}
    def ani_func_2(u):
        return 0.0
    output = ani._evaluate_animation_function(model, ani_func_2, inp_dict)
    assert output == 0.0


def test_animate_with_wrong_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = GridSampler(domain=I, n_points=30)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    with pytest.raises(AssertionError):
        ani.animate(model, ani_func, ps)


def test_animate_with_wrong_output_shape():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=20, density=0.3)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1},
                      solution_dims={'u': 10},
                      width=5, depth=1)
    with pytest.raises(NotImplementedError):
        ani.animate(model, ani_func, ps)


def test_line_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    I2 = Interval(R1('x'), 0, lambda t:t+1)
    ps = AnimationSampler(plot_domain=I2, animation_domain=I,
                          frame_number=10, n_points=10, 
                          dic_for_other_variables={'D': [1, 1.0]})
    model = SimpleFCN(variable_dims={'x': 1, 't': 1, 'D': 2},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          dic_for_other_variables={'D': 0.3})
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=50)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_surface_animation_for_domain_operations():
    I = Interval(R1('t'), 0, 1)
    C = Parallelogram(R2('x'), [-4, -4], [4, -4], [-4, 4]) - \
            Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=2, density=0.8)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_contour_animation():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 3)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=10, 
                          dic_for_other_variables={'D': 0.3})
    model = SimpleFCN(variable_dims={'x': 2, 't': 1, 'D': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
    assert ps.plot_domain_constant


def test_2d_contour_animation_if_domain_changes():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=5, n_points=50)
    model = SimpleFCN(variable_dims={'x': 2, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    assert not ps.plot_domain_constant            
    fig, animation = ani.animate(model, ani_func, ani_sampler=ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)


def test_2d_contour_animation_for_domain_operations():
    I = Interval(R1('t'), 0, 1)
    C = Parallelogram(R1('x')*R1('y'), [-4, -4], [4, -4], [-4, 4]) - \
            Circle(R1('x')*R1('y'), [0, 0], lambda t: t+1)
    ps = AnimationSampler(plot_domain=C, animation_domain=I,
                          frame_number=2, density=0.8)
    model = SimpleFCN(variable_dims={'x': 1, 'y': 1, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps, ani_type='contour_surface')
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)
    assert not ps.plot_domain_constant


def test_line_animation():
    I = Interval(R1('t'), 0, 1)
    I2 = Interval(R1('x'), 0, 1)
    ps = AnimationSampler(plot_domain=I2, animation_domain=I,
                          frame_number=5, n_points=10)
    model = SimpleFCN(variable_dims={'x': 1, 't': 1},
                      solution_dims={'u': 1},
                      width=5, depth=1)
    fig, animation = ani.animate(model, ani_func, ps)
    animation.save('test.gif')
    os.remove('test.gif')
    pyplot.close(fig)