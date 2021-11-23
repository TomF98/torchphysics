import pytest
import torch

from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.domains import (Circle, Interval, Point, Parallelogram)
from torchphysics.problem.samplers import *
from torchphysics.utils.user_fun import UserFunction


def filter_func(x):
    return x[:, 0] <= 0


def test_sampler_creation():
    ps = PointSampler(n_points=40, density=0.3)
    assert ps.n_points == 40
    assert ps.density == 0.3
    assert ps.filter == None


def test_sampler_creation_with_filter():
    ps = PointSampler(n_points=410, density=0.5, filter=lambda t: 2*t)
    assert ps.n_points == 410
    assert ps.density == 0.5
    assert isinstance(ps.filter, UserFunction)


def test_sampler_len_for_n():
    ps = PointSampler(n_points=21)
    assert len(ps) == 21


def test_sampler_set_length():
    ps = PointSampler()
    ps.set_length(34)
    assert len(ps) == 34


def test_sampler_len_for_density_not_definied():
    ps = PointSampler(density=0.4)
    with pytest.raises(ValueError):
        len(ps)


def test_sampler_param_repeat():
    ps = PointSampler(n_points=25)
    test_params = {'t': torch.tensor([[9.0], [4.0]]), 
                   'x': torch.tensor([[1.0, 1.0], [0.0, 2.0]])}
    not_repeated = ps._repeat_input_params(1, **test_params)
    assert test_params == not_repeated
    repeated = ps._repeat_input_params(2, **test_params)
    assert all(repeated['t'] == torch.tensor([[9.0], [9.0], [4.0], [4.0]]))
    assert torch.all(repeated['x'] == \
                     torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 2.0], [0.0, 2.0]]))


def test_sampler_point_repeat():
    ps = PointSampler(n_points=25)
    test_params = {'t': torch.tensor([[9.0], [4.0]]), 
                   'x': torch.tensor([[1.0, 1.0], [0.0, 2.0]])}
    not_repeated = ps._repeat_sampled_points(1, test_params)
    assert test_params == not_repeated
    repeated = ps._repeat_sampled_points(2, test_params)
    assert all(repeated['t'] == torch.tensor([[9.0], [4.0], [9.0], [4.0]]))
    assert torch.all(repeated['x'] == \
                     torch.tensor([[1.0, 1.0], [0.0, 2.0], [1.0, 1.0], [0.0, 2.0]]))


def test_sampler_extracts_length_from_dict():
    ps = PointSampler()
    test_params = {'t': torch.tensor([[9.0], [4.0], [6.7]]), 
                   'x': torch.tensor([[1.0, 1.0], [0.0, 2.0], [9, 3]])}
    num_of_params = ps._extract_tensor_len_from_dict(test_params)  
    assert num_of_params == 3


def test_sampler_extracts_length_from_empty_dict():
    ps = PointSampler()
    num_of_params = ps._extract_tensor_len_from_dict({})  
    assert num_of_params == 1


def test_sampler_extracts_length_from_dict():
    ps = PointSampler()
    test_params = {'t': torch.tensor([[9.0], [4.0], [6.7]]), 
                   'x': torch.tensor([[1.0, 1.0], [0.0, 2.0], [9, 3]])}
    num_of_params = ps._extract_tensor_len_from_dict(test_params)  
    assert num_of_params == 3


def test_sampler_extracts_ith_dict_row():
    ps = PointSampler()
    test_params = {'t': torch.tensor([[9.0], [4.0], [6.7]]), 
                   'x': torch.tensor([[1.0, 1.0], [0.0, 2.0], [9, 3]])}
    row_0 = ps._extract_points_from_dict(0, test_params)
    assert row_0['t'] == 9.0
    assert torch.all(row_0['x'] == torch.tensor([1.0, 1.0])) 
    row_0 = ps._extract_points_from_dict(2, test_params)
    assert row_0['t'] == 6.7
    assert torch.all(row_0['x'] == torch.tensor([9.0, 3.0])) 


def test_sampler_append_dictionarys():
    ps = PointSampler()
    dict_1 = {'t': torch.tensor([[9.0]]), 
              'x': torch.tensor([[1.0, 1.0]])}
    dict_2 = {'t': torch.tensor([[4.0], [6.7]]), 
              'x': torch.tensor([[0.0, 2.0], [9, 3]])}
    ps._append_point_dict(dict_1, dict_2)
    assert torch.all(dict_1['t'] == torch.tensor([[9.0], [4.0], [6.7]]))
    assert torch.all(dict_1['x'] == torch.tensor([[1.0, 1.0], [0.0, 2.0], [9, 3]]))


def test_sampler_apply_filter():
    ps = PointSampler(filter=lambda x: x>=0)
    test_dict = {'x': torch.tensor([[1.0], [-10], [0.1]]), 
                 't': torch.tensor([[1.0, 0.0], [2.3, 2.3], [0.0, 0.0]])}
    n = ps._apply_filter(test_dict)
    assert n == 2
    assert torch.all(test_dict['x'] == torch.tensor([[1.0], [0.1]]))
    assert torch.all(test_dict['t'] == torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_sampler_iteration_check():
    ps = PointSampler()
    ps._check_iteration_number(3, 0)


def test_sampler_iteration_check_warning():
    ps = PointSampler()
    with pytest.warns(None):
        ps._check_iteration_number(10, 10)


def test_sampler_iteration_check_error():
    ps = PointSampler()
    with pytest.raises(RuntimeError):
        ps._check_iteration_number(23, 0)


def test_sampler_product():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    product = ps_1 * ps_2
    assert product.sampler_a == ps_1
    assert product.sampler_b == ps_2


def test_sampler_product_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=10)
    product = ps_1 * ps_2
    assert len(product) == 300


def test_sampler_sum():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    sampler_sum = ps_1 + ps_2
    assert sampler_sum.sampler_a == ps_1
    assert sampler_sum.sampler_b == ps_2


def test_sampler_sum_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=10)
    sampler_sum = ps_1 + ps_2
    assert len(sampler_sum) == 40


def test_sampler_append():
    ps_1 = PointSampler()
    ps_2 = PointSampler()
    sampler_append = ps_1.append(ps_2)
    assert sampler_append.sampler_a == ps_1
    assert sampler_append.sampler_b == ps_2


def test_sampler_append_length():
    ps_1 = PointSampler(n_points=30)
    ps_2 = PointSampler(n_points=30)
    sampler_append = ps_1.append(ps_2)
    assert len(sampler_append) == 30


def test_random_sampler():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, 50)
    points = ps.sample_points()
    assert points['x'].shape == (50, 2)
    assert all(C.__contains__(points))


def test_random_sampler_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, density=0.5)
    points = ps.sample_points()
    assert points['x'].shape == (114, 2)
    assert all(C.__contains__(points))


def test_random_sampler_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = RandomUniformSampler(I, n_points=10)
    ps_C = RandomUniformSampler(C, n_points=20)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 2)
    assert all(C.__contains__(points))


def test_random_sampler_product_with_density():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = RandomUniformSampler(I, n_points=10)
    ps_C = RandomUniformSampler(C, density=0.8)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, density=0.4, filter=filter_func)
    points = ps.sample_points()
    assert torch.all(filter_func(x=points['x']))
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_n_and_without_params():
    C = Circle(R2('x'), [0, 0], 3)
    ps = RandomUniformSampler(C, n_points=20, filter=filter_func)
    points = ps.sample_points()
    assert points['x'].shape == (20, 2)
    assert torch.all(filter_func(x=points['x']))
    assert all(C.__contains__(points))


def test_random_sampler_with_filter_and_n_and_with_params():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    pi = RandomUniformSampler(I, n_points=10)
    ps = RandomUniformSampler(C, n_points=50, filter=filter_func)
    ps *= pi 
    points = ps.sample_points()
    assert points['x'].shape == (500, 2)
    assert points['t'].shape == (500, 1)
    assert torch.all(filter_func(x=points['x']))
    assert all(C.__contains__(points))


def test_grid_sampler():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GridSampler(P, 50)
    points = ps.sample_points()
    assert points['x'].shape == (50, 2)
    assert all(P.__contains__(points))


def test_grid_sampler_density():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GridSampler(P, density=0.5)
    points = ps.sample_points()
    assert points['x'].shape == (8, 2)
    assert all(P.__contains__(points))


def test_grid_sampler_product():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_C = GridSampler(C, n_points=20)
    ps = ps_C * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 2)
    assert all(C.__contains__(points))


def test_grid_sampler_sum():
    P = Point(R2('x'), [0, 0])
    C = Circle(R2('x'), [0, 0], 3)
    ps_1 = GridSampler(P, n_points=10)
    ps_2 = GridSampler(C, n_points=20)
    ps = ps_1 + ps_2
    points = ps.sample_points()
    assert points['x'].shape == (30, 2)
    assert all(C.__contains__(points))


def test_grid_sampler_with_filter_and_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C.boundary, density=0.4, filter=filter_func)
    points = ps.sample_points()
    assert torch.all(filter_func(x=points['x']))
    assert all(C.boundary.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_without_params():
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter=filter_func)
    points = ps.sample_points()
    assert points['x'].shape == (20, 2)
    assert torch.all(filter_func(x=points['x']))
    assert all(C.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_with_params():
    I = Interval(R1('t'), 0, 2)
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    pi = GridSampler(I, n_points=10)
    ps = GridSampler(C, n_points=50, filter=filter_func)
    ps *= pi 
    points = ps.sample_points()
    assert points['x'].shape == (500, 2)
    assert points['t'].shape == (500, 1)
    assert torch.all(filter_func(x=points['x']))
    assert all(C.__contains__(points))


def test_grid_sampler_with_filter_and_n_and_all_points_valid():
    def redudant_filter(x):
        return x[:, 0] <= 10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter=redudant_filter)
    points = ps.sample_points()
    assert points['x'].shape == (20, 2)
    assert torch.all(redudant_filter(x=points['x']))
    assert all(C.__contains__(points))


def test_grid_sampler_with_impossible_filter_and_n():
    def impossible_filter(x):
        return x[:, 0] <= -10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=20, filter=impossible_filter)
    with pytest.raises(RuntimeError):
        _ = ps.sample_points()


def test_grid_sampler_resample_grid_warning():
    def redudant_filter(x):
        return x[:, 0] <= 10
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=10, filter=redudant_filter)
    with pytest.warns(None):
        points = ps._resample_grid({}, 0, ps.domain.sample_grid, {})
    assert points['x'].shape == (110, 2)
    assert torch.all(redudant_filter(x=points['x']))
    assert all(C.__contains__(points))


def test_grid_sampler_append_no_random_points():
    C = Circle(R2('x'), [0, 0], 3)
    ps = GridSampler(C, n_points=10, filter=filter_func)
    assert ps._append_random_points({}, 10, {}) is None


def test_spaced_grid_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = SpacedGridSampler(I, 50, exponent=2)
    points = ps.sample_points()
    assert points['t'].shape == (50, 1)
    assert all(I.__contains__(points))


def test_spaced_grid_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = SpacedGridSampler(P, 50, exponent=2)


def test_spaced_grid_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = SpacedGridSampler(I_2, n_points=20, exponent=3.2)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 1)
    assert all(I_2.__contains__(points))


def test_spaced_grid_sampler_append():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = SpacedGridSampler(I_2, n_points=10, exponent=0.4)
    ps = ps_I.append(ps_I_2)
    points = ps.sample_points()
    assert points['x'].shape == (10, 1)
    assert points['t'].shape == (10, 1)
    assert all(I_2.__contains__(points))
    assert all(I.__contains__(points))


# plot samplers

def test_plot_sampler_creation():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    assert len(ps) == 34
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 3
    assert isinstance(ps.sampler, ConcatSampler)


def test_plot_sampler_creation_only_grid_samplers_used():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    assert isinstance(ps.sampler.sampler_a, GridSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_with_density():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, density=0.4)
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 3
    assert isinstance(ps.sampler, ConcatSampler)


def test_plot_sampler_creation_with_intrval():
    I = Interval(R1('t'), 0, 2)
    ps = PlotSampler(I, n_points=34)
    assert isinstance(ps.sampler, ConcatSampler)
    assert isinstance(ps.sampler.sampler_a, ConcatSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_with_intrval_with_density():
    I = Interval(R1('t'), 0, 2)
    ps = PlotSampler(I, density=0.3)
    assert isinstance(ps.sampler, ConcatSampler)
    assert isinstance(ps.sampler.sampler_a, ConcatSampler)
    assert isinstance(ps.sampler.sampler_b, GridSampler)


def test_plot_sampler_creation_for_variable_domain():
    C = Circle(R2('x'), [0, 0], lambda t: t+1)
    ps = PlotSampler(C, n_points=34, dic_for_other_variables={'t': 1.0, 'D': [9.0, 2.0]})
    assert len(ps) == 34
    assert ps.domain.center.fun == [0, 0]
    assert ps.domain.radius.fun == 2


def test_plot_sampler_transform_dict_to_tensors():
    C = Circle(R2('x'), [0, 0], 1)
    ps = PlotSampler(C, n_points=34)
    ps.dic_for_other_variables = {'x': 3.0, 't': [0, 0.0], 'D': torch.tensor([0.9])}
    tensor_dict = ps._transform_input_dict_to_tensor_dict()
    for key, item in ps.dic_for_other_variables.items():
        assert isinstance(tensor_dict[key], torch.Tensor)
        if isinstance(item, torch.Tensor):
            assert torch.allclose(tensor_dict[key], item)   
        else:
            assert torch.allclose(tensor_dict[key], torch.tensor(item))   


def test_plot_sampler_set_device_and_grad():
    C = Circle(R2('x'), [0, 0], 1)
    ps = PlotSampler(C, n_points=34)
    test_dict = {'t': torch.tensor([9.0]), 'x':torch.tensor([[2, 3.0]])}
    ps._set_device_and_grad_true(test_dict)
    for item in test_dict.values():
        assert item.requires_grad
        assert item.device.type == 'cpu'   


def test_plot_sampler_transform_other_dict_data_to_tensor():
    C = Circle(R2('x'), [0, 0], 1)
    ps = PlotSampler(C, n_points=34)
    ps.dic_for_other_variables = {'x': 3.0, 't': [0, 0.0]}
    test_dict = {}
    ps._add_other_variables(test_dict)
    assert test_dict['x'].shape == (34, 1)
    assert test_dict['t'].shape == (34, 2)


def test_plot_sampler_transform_wrong_data_type():
    C = Circle(R2('x'), [0, 0], 1)
    ps = PlotSampler(C, n_points=34)
    ps.dic_for_other_variables = {'x': C}
    test_dict = {}
    with pytest.raises(TypeError):
        ps._add_other_variables(test_dict)


def test_plot_sampler_create_points():
    C = Circle(R2('x'), [0, 0], 3)
    ps = PlotSampler(C, n_points=34)
    points = ps.sample_points()
    in_C = C._contains(points)
    on_C = C.boundary._contains(points)
    assert all(torch.logical_or(in_C, on_C))


## Test Gaussian sampler

def test_gaussian_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = GaussianSampler(I, 50, mean=0.2, std=0.1)
    points = ps.sample_points()
    assert points['t'].shape == (50, 1)
    assert all(I.__contains__(points))


def test_gaussian_sampler_in_2D():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    ps = GaussianSampler(P, 100, mean=[0.2, 0.3], std=0.1)
    points = ps.sample_points()
    assert points['x'].shape == (100, 2)
    assert all(P.__contains__(points))


def test_gaussian_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P.boundary, 50, mean=[0, 0], std=0.2)


def test_gaussian_sampler_wrong_mean_dimension():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P, 50, mean=torch.tensor([0, 0, 0.3]), std=0.2)


def test_gaussian_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = GaussianSampler(I_2, n_points=20, mean=1, std=0.3)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 1)
    assert points['t'].shape == (200, 1)
    assert all(I_2.__contains__(points))


def test_gaussian_sampler_product_in_2D():
    I = Interval(R1('t'), 0, 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = GaussianSampler(C, n_points=20, mean=[-2, 0], std=0.3)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 2)
    assert points['t'].shape == (200, 1)
    assert all(C.__contains__(points))



## Test LHS sampler

def test_lhs_sampler():
    I = Interval(R1('t'), 0, 1)
    ps = LHSSampler(I, 50)
    points = ps.sample_points()
    assert points['t'].shape == (50, 1)
    assert all(I.__contains__(points))


def test_lhs_sampler_in_2D():
    P = Parallelogram(R2('x'), [0, 0], [1, 0], [0, 1])
    ps = LHSSampler(P, 100)
    points = ps.sample_points()
    assert points['x'].shape == (100, 2)
    assert all(P.__contains__(points))


def test_lhs_sampler_wrong_domain_type():
    P = Parallelogram(R2('x'), [0, 0], [2, 0], [0, 1])
    with pytest.raises(AssertionError):
        _ = GaussianSampler(P.boundary, 50, mean=[0, 0], std=0.2)


def test_lhs_sampler_product():
    I = Interval(R1('t'), 0, 1)
    I_2 = Interval(R1('x'), 0, lambda t : t+1)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = LHSSampler(I_2, n_points=20)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 1)
    assert points['t'].shape == (200, 1)
    assert all(I_2.__contains__(points))


def test_lhs_sampler_product_in_2D():
    I = Circle(R1('t')*R1('y'), [0,0], 1)
    C = Circle(R2('x'), [0, 0], 2)
    ps_I = GridSampler(I, n_points=10)
    ps_I_2 = LHSSampler(C, n_points=20)
    ps = ps_I_2 * ps_I
    points = ps.sample_points()
    assert points['x'].shape == (200, 2)
    assert points['t'].shape == (200, 1)
    assert points['y'].shape == (200, 1)
    assert all(C.__contains__(points))