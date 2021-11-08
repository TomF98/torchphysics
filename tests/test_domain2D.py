import pytest
import torch

from torchphysics.problem.domain.newdomain2D import Circle, CircleBoundary
from torchphysics.problem.space.space import R2

def radius(t):
    return t + 1 

def center(t):
    return torch.column_stack((t, torch.zeros_like(t)))


def test_create_circle():
    C = Circle(R2('x'), [0, 0], 1)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0])))
    assert C.radius.fun == 1
    assert 'x' in C.space


def test_create_circle_with_variable_bounds():
    C = Circle(R2('x'), center, radius)
    assert C.center.fun == center
    assert C.radius.fun == radius


def test_create_circle_mixed_bounds():
    C = Circle(R2('x'), center, 2.0)
    assert C.center.fun == center
    assert C.radius.fun == 2.0
    C = Circle(R2('x'), [0, 0], radius)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0])))
    assert C.radius.fun == radius


def test_call_circle():
    C = Circle(R2('x'), [0, 0], radius)
    called_C = C(t=2)
    assert all(torch.isclose(torch.tensor(called_C.center.fun),
                             torch.tensor([0, 0])))
    assert called_C.radius.fun == 3


def test_bounding_box_circle():
    C = Circle(R2('x'), [1, 0], 4)
    bounds = C.bounding_box()
    assert bounds[0] == -3
    assert bounds[1] == 5
    assert bounds[2] == -4
    assert bounds[3] == 4


def test_bounding_box_circle_variable_params():
    C = Circle(R2('x'), center, 4)
    bounds = C.bounding_box(t=torch.tensor([1, 2, 3, 4]).reshape(-1, 1))
    assert bounds[0] == -3
    assert bounds[1] == 8
    assert bounds[2] == -4
    assert bounds[3] == 4


def test_circle_contains():
    C = Circle(R2('x'), [0, 0], 4)
    points = torch.tensor([[0.0, 0.0], [0, -2], [-0.1, -8], [4.1, 0]])
    inside = C.__contains__({'x': points})
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_contains_if_radius_changes():
    C = Circle(R2('x'), [0, 0], radius)
    points = torch.tensor([[0.0, 0.0], [0, -2], [4.5, -0.1], [4.1, 0], [-0.1, -8]])
    time = torch.tensor([0, 1, 5, 0.1, 1]).reshape(-1, 1)
    inside = C.__contains__({'x': points, 't': time})
    assert all(inside[:3])
    assert not any(inside[3:])


def test_circle_contains_if_both_params_changes():
    C = Circle(R2('x'), center, radius)
    points = torch.tensor([[0.0, 0.0], [1, 1.5], [-0.1, -8], [4.1, 0], [-0.1, -8]])
    time = torch.tensor([0.1, 1, 5, 0.1, 1]).reshape(-1, 1)
    inside = C.__contains__({'x': points, 't': time})
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_random_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_random_uniform(n=40)
    assert points['x'].shape == (40, 2)
    assert all(torch.norm(points['x'], dim=1) <= 4.0)


def test_circle_random_sampling_with_n_and_variable_radius():
    C = Circle(R2('x'), [0, 0], radius)
    points = C.sample_random_uniform(n=4, t=torch.tensor([0, 1]).reshape(-1, 1))
    assert points['x'].shape == (8, 2)
    assert all(C.__contains__(points,
                              t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)))


def test_circle_random_sampling_with_n_and_variable_radius_and_center():
    C = Circle(R2('x'), center, radius)
    points = C.sample_random_uniform(n=4, t=torch.tensor([0, 1]).reshape(-1, 1))
    assert points['x'].shape == (8, 2)
    assert all(C.__contains__(points,
                              t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)))


def test_circle_grid_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4)
    points = C.sample_grid(n=40)
    assert points['x'].shape == (40, 2)
    assert all(torch.norm(points['x'], dim=1) <= 4.0)


def test_circle_grid_sampling_with_n_and_variable_radius_and_center():
    C = Circle(R2('x'), center, radius)
    points = C.sample_grid(n=3, t=torch.tensor([0, 1, 2]).reshape(-1, 1))
    assert points['x'].shape == (9, 2)
    assert all(C.__contains__(points,
                              t=torch.tensor([0, 0, 0, 1, 1, 1, 2.0, 2, 2]).reshape(-1, 1)))


def test_get_circle_boundary():
    C = Circle(R2('x'), [0, 0], 4)
    boundary = C.boundary
    assert isinstance(boundary, CircleBoundary)
    assert C == boundary.domain


def test_call_circle_boundary():
    C = Circle(R2('x'), [0, 0], radius).boundary
    new_C = C(t=2)
    assert isinstance(new_C, CircleBoundary)
    assert new_C.domain.radius.fun == 3
    assert new_C.domain.center.fun[0] == 0
    assert new_C.domain.center.fun[1] == 0


def test_circle_boundary_contains():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = torch.tensor([[0, 4], [0, -4], [-0.1, 0.5], [-1, -5]])
    inside = C.__contains__({'x': points})
    assert all(inside[:2])
    assert not any(inside[2:])


def test_circle_boundary_contains_if_params_change():
    C = Circle(R2('x'), [0, 0], radius).boundary
    points = torch.tensor([[0, 1], [0, -1], [-2, 0], [0, 2], [0, 1], [-1, -5]])
    time = torch.tensor([0, 0, 1, 1, 1, 2.0]).reshape(-1, 1)
    inside = C.__contains__({'x': points, 't': time})
    assert all(inside[:4])
    assert not any(inside[4:])


def test_circle_boundary_random_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_random_uniform(n=10)
    assert points['x'].shape == (10, 2)
    assert all(torch.isclose(torch.norm(points['x'], dim=1), torch.tensor(4.0)))


def test_circle_boundary_random_sampling_with_n_and_variable_domain():
    C = Circle(R2('x'), center, radius).boundary
    points = C.sample_random_uniform(n=4, t=torch.tensor([0.0, 1.0]).reshape(-1, 1))
    assert points['x'].shape == (8, 2)
    assert all(C.__contains__(points, t=torch.tensor([0, 0, 0, 0, 1.0, 1, 1, 1]).reshape(-1, 1)))


def test_circle_boundary_grid_sampling_with_n():
    C = Circle(R2('x'), [0, 0], 4).boundary
    points = C.sample_grid(n=30)
    assert points['x'].shape == (30, 2)
    assert all(torch.isclose(torch.norm(points['x'], dim=1), torch.tensor(4.0)))


def test_circle_boundary_grid_sampling_with_n_and_variable_domain():
    C = Circle(R2('x'), center, radius).boundary
    points = C.sample_grid(n=2, t=torch.tensor([0.0, 1.0, 2.0, 5.5]).reshape(-1, 1))
    assert points['x'].shape == (8, 2)
    time = {'t' : torch.tensor([0, 0, 1.0, 1, 2, 2, 5.5, 5.5]).reshape(-1, 1)}
    assert all(C.__contains__(points, **time))


def test_circle_normals():
    C = Circle(R2('x'), [0, 0], 4).boundary
    normals = C.normal({'x': torch.tensor([[-4.0, 0], [4, 0], [0, 4], [0, -4]])})
    assert normals.shape == (4, 2)
    assert torch.all(torch.isclose(torch.tensor([[-1.0, 0], [1, 0], [0, 1], [0, -1]]),
                                   normals))


def test_circle_normals_if_domain_changes():
    C = Circle(R2('x'), center, radius).boundary
    normals = C.normal({'x': torch.tensor([[1, 0], [1, 2], [2.0, -3.0]]),
                        't': torch.tensor([0, 1, 2.0]).reshape(-1, 1)})
    assert normals.shape == (3, 2)
    assert torch.all(torch.isclose(torch.tensor([[1.0, 0], [0, 1], [0, -1]]), normals))