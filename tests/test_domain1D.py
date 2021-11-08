import pytest
import torch

from torchphysics.problem.domain.newdomain1D import (Interval,
                                                     IntervalBoundary, 
                                                     IntervalSingleBoundaryPoint)
from torchphysics.problem.space.space import R1


def lower_bound(t):
    return -t

def upper_bound(t):
    return 2*t + 1


def test_create_interval():
    I = Interval(R1('x'), 0, 1)
    assert I.lower_bound.fun == 0
    assert I.upper_bound.fun == 1
    assert 'x' in I.space


def test_create_interval_with_variable_bounds():
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=upper_bound)
    assert I.lower_bound.fun == lower_bound
    assert I.upper_bound.fun == upper_bound


def test_create_interval_mixed_bounds():
    I = Interval(R1('x'), lower_bound=0, upper_bound=upper_bound)
    assert I.lower_bound.fun == 0
    assert I.upper_bound.fun == upper_bound
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=1)
    assert I.lower_bound.fun == lower_bound
    assert I.upper_bound.fun == 1


def test_call_interval():
    I = Interval(R1('x'), lower_bound=0, upper_bound=upper_bound)
    called_I = I(t=2)
    assert called_I.lower_bound.fun == 0
    assert called_I.upper_bound.fun == 5


def test_bounding_box_interval():
    I = Interval(R1('x'), 0, 1)
    bounds = I.bounding_box()
    assert bounds[0] == 0
    assert bounds[1] == 1


def test_bounding_box_interval_variable_bounds():
    I = Interval(R1('x'), lower_bound=lower_bound, upper_bound=upper_bound)
    bounds = I.bounding_box(t=torch.tensor([1, 2, 3, 4]).reshape(-1, 1))
    assert bounds[0] == -4
    assert bounds[1] == 9


def test_interval_contains():
    I = Interval(R1('x'), 0, 1)
    points = torch.tensor([0.5, 0.7, 0, -2, -0.1]).reshape(-1, 1)
    inside = I.__contains__({'x': points})
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_contains_if_one_bound_changes():
    I = Interval(R1('x'), 0, upper_bound)
    points = torch.tensor([0.5, 0, 7, -2, -0.1]).reshape(-1, 1)
    time = torch.tensor([0, 0, 1, -2, -0.1]).reshape(-1, 1)
    inside = I.__contains__({'x': points, 't': time})
    assert all(inside[:2])
    assert not any(inside[2:])


def test_interval_contains_if_both_bound_changes():
    I = Interval(R1('x'), lower_bound, upper_bound)
    points = torch.tensor([0.5, -1, 7, -2, -0.1]).reshape(-1, 1)
    time = torch.tensor([0, 2, 1, -2, -0.1]).reshape(-1, 1)
    inside = I.__contains__({'x': points, 't': time})
    assert all(inside[:2])
    assert not any(inside[2:])


def test_interval_random_sampling_with_n():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_random_uniform(n=10)
    assert points['x'].shape == (10, 1)
    assert all(I.__contains__(points))


def test_interval_random_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound)
    points = I.sample_random_uniform(n=4, t=torch.tensor([0, 1]).reshape(-1, 1))
    assert points['x'].shape == (8, 1)
    assert all(I.__contains__(points, t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)))


def test_interval_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 1)
    points = I.sample_grid(n=10)
    assert points['x'].shape == (10, 1)
    assert all(I.__contains__(points))
    for i in range(8):
        dist_1 = torch.norm(points['x'][i+1] - points['x'][i])
        dist_2 = torch.norm(points['x'][i+1] - points['x'][i+2])
        assert torch.isclose(dist_1, dist_2)


def test_interval_grid_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound)
    points = I.sample_grid(n=4, t=torch.tensor([0, 1]).reshape(-1, 1))
    assert points['x'].shape == (8, 1)
    assert all(I.__contains__(points, t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)))


def test_get_Intervalboundary():
    I = Interval(R1('x'), 0, 1)
    boundary = I.boundary
    assert isinstance(boundary, IntervalBoundary)
    assert I == boundary.domain


def test_call_interval_boundary():
    I = Interval(R1('x'), 0, upper_bound).boundary
    new_I = I(t=2)
    assert isinstance(new_I, IntervalBoundary)
    assert new_I.domain.lower_bound.fun == 0
    assert new_I.domain.upper_bound.fun == 5


def test_interval_boundary_contains():
    I = Interval(R1('x'), 0, 1).boundary
    points = torch.tensor([0, 0, 1, -2, -0.1, 0.5]).reshape(-1, 1)
    inside = I.__contains__({'x': points})
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_boundary_contains_if_bound_changes():
    I = Interval(R1('x'), 0, upper_bound).boundary
    points = torch.tensor([0, 1, 0, 4, -1, 12.0]).reshape(-1, 1)
    time = torch.tensor([0, 0, 1, 1, 1, 2.0]).reshape(-1, 1)
    inside = I.__contains__({'x': points, 't': time})
    assert all(inside[:3])
    assert not any(inside[3:])


def test_interval_boundary_random_sampling_with_n():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_random_uniform(n=10)
    assert points['x'].shape == (10, 1)
    assert all(I.__contains__(points))


def test_interval_boundary_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 1).boundary
    points = I.sample_grid(n=10)
    assert points['x'].shape == (10, 1)
    assert all(I.__contains__(points))
    for i in range(10):
        point_eq_0 = points['x'][i] == 0
        point_eq_1 = points['x'][i] == 1
        assert point_eq_0 or point_eq_1 


def test_interval_boundary_random_sampling_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound).boundary
    points = I.sample_random_uniform(n=2, t=torch.tensor([0.0, 1.0, 2]).reshape(-1, 1))
    assert points['x'].shape == (6, 1)
    assert all(I.__contains__(points, t=torch.tensor([0, 0, 1.0, 1, 2, 2]).reshape(-1, 1)))


def test_interval_boundary_grid_with_n_and_variable_bounds():
    I = Interval(R1('x'), 0, upper_bound).boundary
    points = I.sample_grid(n=4, t=torch.tensor([0.0, 1.0]).reshape(-1, 1))
    assert points['x'].shape == (8, 1)
    assert all(I.__contains__(points, t=torch.tensor([0, 0, 0, 0, 1.0, 1, 1, 1]).reshape(-1, 1)))


def test_interval_normals():
    I = Interval(R1('x'), 0, 1).boundary
    normals = I.normal({'x': torch.tensor([0, 1.0, 0]).reshape(-1, 1)})
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[-1], [1], [-1]]), normals))


def test_interval_normals_if_bounds_change():
    I = Interval(R1('x'), lower_bound, 1).boundary
    normals = I.normal({'x': torch.tensor([0, 1.0, -1, 1.0, -2]).reshape(-1, 1),
                        't': torch.tensor([0, 0, 1, 1, 2.0]).reshape(-1, 1)})
    assert normals.shape == (5, 1)
    assert all(torch.isclose(torch.tensor([[-1], [1], [-1], [1], [-1]]), normals))


def test_interval_get_left_boundary():
    I = Interval(R1('x'), 0, 1).boundary_left
    assert isinstance(I, IntervalSingleBoundaryPoint)
    assert I.side.fun == 0


def test_interval_get_right_boundary():
    I = Interval(R1('x'), 0, 1).boundary_right
    assert isinstance(I, IntervalSingleBoundaryPoint)
    assert I.side.fun == 1


def test_call_single_interval_bound():
    I = Interval(R1('x'), 0, upper_bound).boundary_right
    called_I = I(t=3)
    assert isinstance(called_I, IntervalSingleBoundaryPoint)
    assert called_I.side.fun == upper_bound


def test_single_interval_bound_contains():
    I = Interval(R1('x'), 0, 4).boundary_right
    points = {'x': torch.tensor([[4.0], [0.0], [3.9]])}
    inside = I.__contains__(points)
    assert inside[0]
    assert not any(inside[1:])


def test_single_interval_bound_contains_if_bound_variable():
    I = Interval(R1('x'), 0, upper_bound).boundary_right
    points = {'x': torch.tensor([[5.0], [1.0], [3.9]]), 
              't': torch.tensor([[2.0], [0.0], [-1.0]])}
    inside = I.__contains__(points)
    assert inside.shape == (3, 1)
    assert all(inside[:2])
    assert not any(inside[2])   


def test_single_interval_bound_bounding_box():
    I = Interval(R1('x'), 0, 4).boundary_right
    bounds = I.bounding_box()
    assert len(bounds) == 2
    assert bounds[0] == 0
    assert bounds[1] == 4


def test_single_interval_bound_random_sampling_with_n():
    I = Interval(R1('x'), 0, 4).boundary_left
    points = I.sample_random_uniform(n=25)
    assert points['x'].shape == (25, 1)
    assert all(torch.isclose(points['x'], torch.tensor(0.0)))


def test_single_interval_bound_random_sampling_with_n_moving_bound():
    I = Interval(R1('x'), lower_bound, 4).boundary_left
    points = I.sample_random_uniform(n=25, t=torch.tensor([[1.0], [0.0]]))
    assert points['x'].shape == (50, 1)
    assert all(torch.isclose(points['x'][:25], torch.tensor(-1.0)))
    assert all(torch.isclose(points['x'][25:], torch.tensor(0.0)))


def test_single_interval_bound_grid_sampling_with_n():
    I = Interval(R1('x'), 0, 4).boundary_left
    points = I.sample_grid(n=25)
    assert points['x'].shape == (25, 1)
    assert all(torch.isclose(points['x'], torch.tensor(0.0)))


def test_interval_normals_left_side():
    I = Interval(R1('x'), 0, 1).boundary_left
    normals = I.normal({'x': torch.tensor([0, 0.0, 0]).reshape(-1, 1)})
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[-1], [-1.0], [-1]]), normals))


def test_interval_normals_ride_side():
    I = Interval(R1('x'), 0, 1).boundary_right
    normals = I.normal({'x': torch.tensor([1, 1.0, 1]).reshape(-1, 1)})
    assert normals.shape == (3, 1)
    assert all(torch.isclose(torch.tensor([[1], [1.0], [1]]), normals))