"""This file contains some sample functions for the domain operations.
Since Union/Cut/Intersection follow the same idea for sampling for a given number of points.
"""
import torch

from torchphysics.problem.spaces.points import Points


def _inside_random_with_n(main_domain, domain_a, domain_b, n, params, invert):
    """Creates a random uniform points inside of a cut or intersection domain.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    invert : bool
        Says if the points should lay in the domain_b (intersection) or if
        not (cut). For the Cut-Domain it is invert=True.
    """
    if n == 1: 
        return _random_points_if_n_eq_1(main_domain, domain_a, domain_b,
                                        params, invert)
    return _random_points_inside(main_domain, domain_a, domain_b, n, params, invert)


def _random_points_if_n_eq_1(main_domain, domain_a, domain_b, params, invert):
    final_points = torch.zeros((len(params), main_domain.dim))
    found_valid = torch.zeros((len(params), 1), dtype=bool) 
    while not all(found_valid):
        new_points = domain_a.sample_random_uniform(n=1, params=params)
        index_valid = _check_in_b(domain_b, params, invert, new_points)
        found_valid[index_valid] = True
        final_points[index_valid] = new_points.as_tensor[index_valid]
    return Points(final_points, main_domain.space)


def _random_points_inside(main_domain, domain_a, domain_b, n, params, invert):
    num_of_params = max(len(params), 1)
    random_points = Points.empty()
    for i in range(num_of_params):
        ith_params = params[i, :]
        number_valid = 0
        scaled_n = n
        while number_valid < n:
            new_points = domain_a.sample_random_uniform(n=int(scaled_n), params=ith_params)
            _, repeat_params = main_domain._repeat_params(len(new_points), ith_params)
            index_valid = _check_in_b(domain_b, repeat_params, invert, new_points)
            number_valid = len(index_valid)
            scaled_n = 5*scaled_n if number_valid == 0 else scaled_n**2/number_valid + 1
        random_points = random_points | new_points[index_valid[:n], ]
    return random_points


def _inside_grid_with_n(main_domain, domain_a, domain_b, n, params, invert):
    """Creates a point grid inside of a cut or intersection domain.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    invert : bool
        Says if the points should lay in the domain_b (intersection) or if
        not (cut). For the Cut-Domain it is invert=True.
    """
    # first sample grid inside the domain_a
    grid_a = domain_a.sample_grid(n=n, params=params)
    _, repeat_params = main_domain._repeat_params(n, params)
    index_valid = _check_in_b(domain_b, repeat_params, invert, grid_a)
    number_inside = len(index_valid)
    if number_inside == n:
        return grid_a
    # if the grid does not fit, scale the number of points
    scaled_n = int(n**2 / number_inside)
    grid_a = domain_a.sample_grid(n=scaled_n, params=params)
    _, repeat_params = main_domain._repeat_params(n, params)
    index_valid = _check_in_b(domain_b, repeat_params, invert, grid_a)
    grid_a = grid_a[index_valid, ]
    if len(grid_a) >= n:
        return grid_a[:n, ] 
    # add some random ones if still some missing
    rand_points = _random_points_inside(main_domain, domain_a, domain_b,
                                        n-len(grid_a), params, invert)
    return grid_a | rand_points


def _check_in_b(domain_b, params, invert, grid_a):
    #check what points are correct
    inside_b = domain_b._contains(grid_a, params)
    if invert:
      inside_b = torch.logical_not(inside_b)
    index = torch.where(inside_b)[0]
    return index


def _boundary_random_with_n(main_domain, domain_a, domain_b, n, params):
    pass


def _boundary_grid_with_n(main_domain, domain_a, domain_b, n, params):
    """Creates a point grid on the boundary of a domain operation.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    params : Points
        Additional parameters for the domains.
    """
    # first sample a grid on both boundaries
    grid_a = domain_a.boundary.sample_grid(n=n, params=params)
    grid_b = domain_b.boundary.sample_grid(n=n, params=params)  
    # check how many points are on the boundary of the operation domain
    on_bound_a, on_bound_b, a_correct, b_correct = \
        _check_points_on_main_boundary(main_domain, grid_a, grid_b, params)
    sum_of_correct = a_correct + b_correct
    if sum_of_correct == n:
        return grid_a[on_bound_a, ] | grid_b[on_bound_b, ]
    # scale the n so that more or fewer points are sampled and try again
    # to get a better grid. For the scaling we approximate the volume of the 
    # the main domain.
    a_surface = domain_a.boundary.volume(params)
    b_surface = domain_b.boundary.volume(params)
    approx_surface = a_surface * a_correct / n + b_surface * b_correct / n
    scaled_a = int(n * a_surface / approx_surface) + 1 # round up  
    scaled_b = max(int(n * b_surface / approx_surface), 1) # round to floor, but not 0
    grid_a = domain_a.boundary.sample_grid(n=scaled_a, params=params)
    grid_b = domain_b.boundary.sample_grid(n=scaled_b, params=params)  
    # check again how what points are correct and now just stay with this grid
    # if still some points are missing add random ones.
    on_bound_a, on_bound_b, a_correct, b_correct = \
        _check_points_on_main_boundary(main_domain, grid_a, grid_b, params)
    final_grid = grid_a[on_bound_a, ] | grid_b[on_bound_b, ]
    if a_correct + b_correct >= n:
        return final_grid[:n, ]
    return final_grid # add random points


def _check_points_on_main_boundary(main_domain, grid_a, grid_b, params):
    _, repeat_params = main_domain._repeat_params(len(grid_a), params)
    on_bound_a = torch.where(main_domain._contains(grid_a, params=repeat_params))[0]
    _, repeat_params = main_domain._repeat_params(len(grid_b), params)
    on_bound_b = torch.where(main_domain._contains(grid_b, params=repeat_params))[0]
    a_correct = len(on_bound_a)
    b_correct = len(on_bound_b)
    return on_bound_a,on_bound_b, a_correct, b_correct