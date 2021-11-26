import pytest
import torch

from torchphysics.problem.domains.domain3D.sphere import (Sphere, SphereBoundary)
from torchphysics.problem.spaces.space import R3
from torchphysics.problem.spaces.points import Points


def radius(t):
    return t + 1 

def center(t):
    return torch.column_stack((t, torch.zeros_like(t), torch.zeros_like(t)))


def test_create_sphere():
    C = Sphere(R3('x'), [0, 0, 0], 1)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0, 0])))
    assert C.radius.fun == 1
    assert 'x' in C.space


def test_create_sphere_with_variable_bounds():
    C = Sphere(R3('x'), center, radius)
    assert C.center.fun == center
    assert C.radius.fun == radius


def test_create_sphere_mixed_bounds():
    C = Sphere(R3('x'), center, 2.0)
    assert C.center.fun == center
    assert C.radius.fun == 2.0
    C = Sphere(R3('x'), [0, 0, 0], radius)
    assert all(torch.isclose(torch.tensor(C.center.fun), torch.tensor([0, 0, 0])))
    assert C.radius.fun == radius


def test_call_sphere():
    C = Sphere(R3('x'), [0, 0, 0], radius)
    called_C = C(t=2)
    assert all(torch.isclose(torch.tensor(called_C.center.fun),
                             torch.tensor([0, 0, 0])))
    assert called_C.radius.fun == 3


def test_bounding_box_sphere():
    C = Sphere(R3('x'), [1, 0, 0], 4)
    bounds = C.bounding_box()
    assert bounds[0] == -3
    assert bounds[1] == 5
    assert bounds[2] == -4
    assert bounds[3] == 4
    assert bounds[4] == -4
    assert bounds[5] == 4


def test_bounding_box_sphere_variable_params():
    C = Sphere(R3('x'), center, 4)
    bounds = C.bounding_box(t=torch.tensor([1, 2, 3, 4]).reshape(-1, 1))
    assert bounds[0] == -3
    assert bounds[1] == 8
    assert bounds[2] == -4
    assert bounds[3] == 4
    assert bounds[4] == -4
    assert bounds[5] == 4


def test_sphere_contains():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    points = torch.tensor([[0.0, 0.0, 0.0], [0, -2, 1], [-0.1, -8, 0], [4.1, 0, 1]])
    points = Points(points, R3('x'))
    inside = C._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_sphere_contains_if_radius_changes():
    C = Sphere(R3('x'), [0, 0, 0], radius)
    points = torch.tensor([[0.0, 0.0, 0.0], [0, -0.5, -1], [4.5, -0.1, 0],
                           [4.1, 0, 10], [-0.1, -8, 0]])
    time = torch.tensor([0, 1, 6, 0.1, 1]).reshape(-1, 1)
    points = Points(points, R3('x'))
    inside = C._contains(points, t=time)
    assert all(inside[:3])
    assert not any(inside[3:])


def test_sphere_contains_if_both_params_changes():
    C = Sphere(R3('x'), center, radius)
    points = torch.tensor([[0.0, 0.0, 0.4], [1, 1.5, 0.0], [-0.1, -8, 0.0], 
                           [4.1, 0, 5], [-0.1, -8, 23]])
    time = torch.tensor([0.1, 1, 5, 0.1, 1]).reshape(-1, 1)
    points = Points(points, R3('x'))
    inside = C._contains(points, t=time)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_sphere_random_sampling_with_n():
    C = Sphere(R3('x'), [0, 0, 0.0], 4)
    points = C.sample_random_uniform(n=40)
    assert points[:, 'x'].shape == (40, 3)
    assert all(torch.linalg.norm(points[:, 'x'], dim=1) <= 4.0)


def test_sphere_random_sampling_with_d():
    C = Sphere(R3('x'), [0, 0, 0.0], 5)
    points = C.sample_random_uniform(d=0.3)
    assert all(torch.linalg.norm(points[:, 'x'], dim=1) <= 5.0)


def test_sphere_random_sampling_with_n_and_variable_radius():
    C = Sphere(R3('x'), [0, 0, 3], radius)
    points = C.sample_random_uniform(n=4, t=torch.tensor([0, 1]).reshape(-1, 1))
    assert points[:, 'x'].shape == (8, 3)
    assert all(C._contains(points,
                           t=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)))


def test_sphere_random_sampling_with_n_and_variable_radius_and_center():
    C = Sphere(R3('x'), center, radius)
    time = torch.tensor([0, 1]).reshape(-1, 1)
    points = C.sample_random_uniform(n=10, t=time)
    assert points[:, 'x'].shape == (20, 3)
    assert all(C._contains(points, t=time.repeat_interleave(10).reshape(-1, 1)))


def test_sphere_grid_sampling_with_n():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    points = C.sample_grid(n=40)
    assert points[:, 'x'].shape == (40, 3)
    assert all(torch.linalg.norm(points[:, 'x'], dim=1) <= 4.0)


def test_sphere_grid_sampling_with_to_small_n():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    points = C.sample_grid(n=4)
    assert points[:, 'x'].shape == (4, 3)
    assert all(torch.linalg.norm(points[:, 'x'], dim=1) <= 4.0)


def test_sphere_append_random():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    points = torch.tensor([[2.0], [0.0]])
    new_points = C._append_random(points, 2, None)
    assert torch.equal(points, new_points)


def test_sphere_grid_sampling_with_d():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    points = C.sample_grid(d=1)
    assert all(torch.linalg.norm(points[:, 'x'], dim=1) <= 4.0)


def test_sphere_grid_sampling_with_n_and_variable_radius_and_center():
    C = Sphere(R3('x'), center, radius)
    points = C.sample_grid(n=50, t=torch.tensor([0.0]).reshape(-1, 1))
    assert points[:, 'x'].shape == (50, 3)
    assert all(C._contains(points,
                           t=torch.tensor([[0.0]]).repeat_interleave(50)))


def test_get_sphere_boundary():
    C = Sphere(R3('x'), [0, 0, 0], 4)
    boundary = C.boundary
    assert isinstance(boundary, SphereBoundary)
    assert C == boundary.domain


def test_call_sphere_boundary():
    C = Sphere(R3('x'), [0, 0, 2], radius).boundary
    new_C = C(t=2)
    assert isinstance(new_C, SphereBoundary)
    assert new_C.domain.radius.fun == 3
    assert new_C.domain.center.fun[0] == 0
    assert new_C.domain.center.fun[1] == 0
    assert new_C.domain.center.fun[2] == 2


def test_sphere_boundary_contains():
    C = Sphere(R3('x'), [0, 0, 0], 4).boundary
    points = torch.tensor([[0, 4, 0], [0, 0, -4], [-0.1, 0.5, 2], [-1, -5, 0]])
    points = Points(points, R3('x'))
    inside = C._contains(points)
    assert all(inside[:2])
    assert not any(inside[2:])


def test_sphere_boundary_contains_if_params_change():
    C = Sphere(R3('x'), [0, 0, 0], radius).boundary
    points = torch.tensor([[0, 1, 0], [0, 0, -1], [-2, 0, 0], [0, 2, 0],
                           [0, 1, 0], [-1, -5, -12]])
    time = torch.tensor([0, 0, 1, 1, 1, 2.0]).reshape(-1, 1)
    points = Points(points, R3('x'))
    inside = C._contains(points, t=time)
    assert all(inside[:4])
    assert not any(inside[4:])


def test_sphere_boundary_random_sampling_with_n():
    C = Sphere(R3('x'), [0, 0, 0], 2).boundary
    points = C.sample_random_uniform(n=10)
    assert points[:, 'x'].shape == (10, 3)
    assert all(torch.isclose(torch.linalg.norm(points[:, 'x'], dim=1), torch.tensor(2.0)))


def test_sphere_boundary_random_sampling_with_d():
    C = Sphere(R3('x'), [0, 0, 0], 4).boundary
    points = C.sample_random_uniform(d=0.5)
    assert all(torch.isclose(torch.linalg.norm(points[:, 'x'], dim=1), torch.tensor(4.0)))


def test_sphere_boundary_random_sampling_with_n_and_variable_domain():
    C = Sphere(R3('x'), center, radius).boundary
    points = C.sample_random_uniform(n=4, t=torch.tensor([0.0, 1.0]).reshape(-1, 1))
    assert points[:, 'x'].shape == (8, 3)
    assert all(C._contains(points, t=torch.tensor([0, 0, 0, 0, 1.0, 1, 1, 1]).reshape(-1, 1)))


def test_sphere_boundary_grid_sampling_with_n():
    C = Sphere(R3('x'), [0, 0, 0], 1).boundary
    points = C.sample_grid(n=30)
    assert points[:, 'x'].shape == (30, 3)
    assert all(torch.isclose(torch.linalg.norm(points[:, 'x'], dim=1), torch.tensor(1.0)))


def test_sphere_boundary_grid_sampling_with_d():
    C = Sphere(R3('x'), [0, 0, 0], 4).boundary
    points = C.sample_grid(d=0.3)
    assert all(torch.isclose(torch.linalg.norm(points[:, 'x'], dim=1), torch.tensor(4.0)))


def test_sphere_boundary_grid_sampling_with_n_and_variable_domain():
    C = Sphere(R3('x'), center, radius).boundary
    points = C.sample_grid(n=20, t=torch.tensor([0.0]).reshape(-1, 1))
    assert points[:, 'x'].shape == (20, 3)
    time = {'t' : torch.tensor([0.0]).repeat_interleave(20).reshape(-1, 1)}
    assert all(C._contains(points, **time))


def test_sphere_normals():
    C = Sphere(R3('x'), [0, 0, 0], 4).boundary
    points = Points(torch.tensor([[-4.0, 0, 0], [4, 0, 0],
                                  [0, 0, 4], [0, -4, 0]]), R3('x'))
    normals = C.normal(points)
    assert normals.shape == (4, 3)
    assert torch.all(torch.isclose(torch.tensor([[-1.0, 0, 0], [1, 0, 0], 
                                                 [0, 0, 1], [0, -1, 0]]),
                                   normals))


def test_sphere_normals_if_domain_changes():
    C = Sphere(R3('x'), center, radius).boundary
    points = Points(torch.tensor([[1, 0, 0], [1, 2, 0], [2.0, -3.0, 0]]), R3('x'))
    normals = C.normal(points,
                       t=torch.tensor([0, 1, 2.0]).reshape(-1, 1))
    assert normals.shape == (3, 3)
    assert torch.all(torch.isclose(torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, -1, 0]]),
                                   normals))

