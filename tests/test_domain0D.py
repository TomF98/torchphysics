import pytest
import torch
import numpy as np

from torchphysics.problem.domain.domain0D.point import Point
from torchphysics.problem.space.space import R1, R2, R3


def p(t):
    return 2*t


def p2(t):
    if not isinstance(t, torch.Tensor):
        return [2*t, 0]
    return torch.column_stack((2*t, torch.zeros_like(t)))


def test_create_point():
    P = Point(R1('x'), 4)
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun == 4


def test_create_point_in_higher_dim():
    P = Point(R2('x'), [4, 0])
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun[0] == 4
    assert P.point.fun[1] == 0


def test_create_point_with_variable_point():
    P = Point(R1('x'), p)
    assert P.dim == 0
    assert 'x' in P.space
    assert P.point.fun == p


def test_call_point():
    P = Point(R2('x'), p2)
    called_P = P(t=3)
    assert called_P.dim == 0
    assert 'x' in called_P.space
    assert called_P.point.fun[0] == 6
    assert called_P.point.fun[1] == 0


def test_point_has_no_boundary():
    P = Point(R2('x'), p2)
    with pytest.raises(NotImplementedError):
        P.boundary


def test_point_contains():
    P = Point(R1('x'), 4)
    points = {'x': torch.tensor([[4.0], [0.0], [3.9]])}
    inside = P._contains(points)
    assert inside[0]
    assert not any(inside[1:])


def test_point_contains_if_point_variable():
    P = Point(R1('x'), p)
    points = {'x': torch.tensor([[4.0], [0.0], [3.9]]), 
              't': torch.tensor([[2.0], [0.0], [-1.0]])}
    inside = P._contains(points)
    assert inside.shape == (3, 1)
    assert all(inside[:2])
    assert not any(inside[2])   


def test_point_contains_in_higher_dim():
    P = Point(R1('x')*R2('y'), [4.0, 0.0, 3.0])
    points = {'x': torch.tensor([[4.0], [0.0], [3.9], [4.0]]), 
              'y': torch.tensor([[0.0, 3.0], [0.0, 3.0], [2.0, 8.0], [1.0, 3.0]])}
    inside = P._contains(points)
    assert inside.shape == (4, 1)
    assert inside[0]
    assert not any(inside[1:])


def test_point_contains_higher_dim_and_variable():
    P = Point(R2('x'), p2)
    points = {'x': torch.tensor([[1.0, 0.0], [3.0, 1.0], [-1.0, 0.0], [2.0, 5.0]])}
    inside = P._contains(points, t=torch.tensor([[1/2.0], [1.0], [0.0], [1.0]]))
    assert inside[0]
    assert not any(inside[1:])


def test_point_bounding_box():
    P = Point(R1('x'), 4)
    bounds = P.bounding_box()
    assert len(bounds) == 2
    assert bounds[0] == 3.9
    assert bounds[1] == 4.1


def test_point_bounding_box_higher_dim():
    P = Point(R1('x')*R2('y'), [4, 3, 1])
    bounds = P.bounding_box()
    assert len(bounds) == 6
    assert bounds[0] == 3.9
    assert bounds[1] == 4.1
    assert bounds[2] == 2.9
    assert bounds[3] == 3.1
    assert bounds[4] == 0.9
    assert bounds[5] == 1.1


def test_point_bounding_box_moving_point():
    P = Point(R1('x'), p)
    bounds = P.bounding_box(t=torch.tensor([[1.0], [9.0], [2]]))
    assert len(bounds) == 2
    assert bounds[0] == 2
    assert bounds[1] == 18


def test_point_bounding_box_moving_point_higher_dim():
    P = Point(R2('x'), p2)
    bounds = P.bounding_box(t=torch.tensor([[1.0], [9.0], [2]]))
    assert len(bounds) == 4
    assert bounds[0] == 2
    assert bounds[1] == 18
    assert np.isclose(bounds[2], -0.1)
    assert np.isclose(bounds[3], 0.1)


def test_point_random_sampling_with_n():
    P = Point(R1('x'), 4)
    points = P.sample_random_uniform(n=25)
    assert points['x'].shape == (25, 1)
    assert all(torch.isclose(points['x'], torch.tensor(4.0)))


def test_point_random_sampling_with_nigher_dim():
    P = Point(R3('x'), [4.0, 0.0, 1.3])
    points = P.sample_random_uniform(n=25)
    assert points['x'].shape == (25, 3)
    assert all(torch.isclose(points['x'][:, 0], torch.tensor(4.0)))
    assert all(torch.isclose(points['x'][:, 1], torch.tensor(0.0)))
    assert all(torch.isclose(points['x'][:, 2], torch.tensor(1.3)))


def test_point_random_sampling_with_n_moving_point():
    P = Point(R1('x'), p)
    points = P.sample_random_uniform(n=25, t=torch.tensor([[1.0], [0.0]]))
    assert points['x'].shape == (50, 1)
    assert all(torch.isclose(points['x'][:25], torch.tensor(2.0)))
    assert all(torch.isclose(points['x'][25:], torch.tensor(0.0)))


def test_point_random_sampling_with_n_moving_point_higher_dim():
    P = Point(R2('x'), p2)
    points = P.sample_random_uniform(n=5, t=torch.tensor([[1.0], [0.0]]))
    assert points['x'].shape == (10, 2)
    assert all(torch.isclose(points['x'][:5, 0], torch.tensor(2.0)))
    assert all(torch.isclose(points['x'][5:, 0], torch.tensor(0.0)))
    assert all(torch.isclose(points['x'][:, 1], torch.tensor(0.0)))


def test_point_grid_sampling_with_n():
    P = Point(R1('x'), 4)
    points = P.sample_grid(n=25)
    assert points['x'].shape == (25, 1)
    assert all(torch.isclose(points['x'], torch.tensor(4.0)))