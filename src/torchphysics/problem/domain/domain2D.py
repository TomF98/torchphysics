import numpy as np
import shapely
import shapely.geometry as s_geo
import shapely.ops as s_ops
import torch

from .domain import Domain, LambdaDomain, BoundaryDomain


class Domain2D(Domain):
    def __init__(self, space, dim):
        if space.dim != 2:
            raise ValueError("""A 2d Domain can only be created in a 2d space.""")
        super().__init__(space, dim=dim)


class Circle(Domain2D):
    """Class for circles.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the circle, e.g. center = [5,0].
    radius : number or callable
        The radius of the circle.
    resolution : number, optional
        The resolution that should be used to approximate the circle with a polygon.
        The number of used vertices is around 4*resolution.
        See shapely for exact informations.
    """
    def __new__(cls, space, center, radius, resolution=10, tol=1e-06):
        if callable(center) or callable(radius):
            params = {'center': center, 'radius': radius, 'resolution': resolution}
            return LambdaDomain(class_=cls, params=params, space=space, dim=2, tol=tol)
        return super(Circle, cls).__new__(cls)

    def __init__(self, space, center, radius):
        self.space = space
        self.center = UserFun(center)
        self.radius = UserFun(radius)
        super().__init__(space=space)

    def contains(self, points, **other_vars):
        return torch.norm(self.space.as_tensor(points)-self.center(**other_vars)) < self.radius(**other_vars)


class CircleBoundary(BoundaryDomain):
    pass


class Domain2D(Domain):

    def __init__(self, polygon, space, tol):
        self.polygon = s_geo.polygon.orient(polygon)
        super().__init__(space, dim=2, tol=tol)

    def __add__(self, other):
        assert other.dim == 2
        new_poly = s_ops.unary_union([self.polygon, other.polygon])
        return Polygon(space=self.space, vertices=new_poly)

    def __sub__(self, other):
        assert other.dim == 2
        new_poly = self.polygon - other.polygon
        return Polygon(space=self.space, vertices=new_poly)

    def __and__(self, other):
        assert other.dim == 2
        new_poly = self.polygon & other.polygon
        return Polygon(space=self.space, vertices=new_poly)

    def outline(self):
        """Creates a outline of the domain.

        Returns
        -------
        list of list
            The vertices of the domain. Inner vertices are appended in there
            own list.
        """
        cords = [np.array(self.polygon.exterior.coords)]
        for i in self.polygon.interiors:
            cords.append(np.array(i.coords))
        return cords

    def is_inside(self, points):
        points = super()._check_single_point(points)
        points = super().return_space_variables_to_point_list(points)
        inside = np.empty(len(points), dtype=bool)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            inside[i] = self.polygon.contains(point)
        return inside.reshape(-1, 1)

    def bounding_box(self):
        bounds = self.polygon.bounds
        return [bounds[0], bounds[2], bounds[1], bounds[3]]

    def sample_random_uniform(self, n):
        points = np.empty((0, self.dim))
        big_t, t_area = None, 0
        # instead of using a bounding box it is more efficient to triangulate
        # the polygon and sample in each triangle.
        for t in s_ops.triangulate(self.polygon):
            new_points = self._sample_in_triangulation(t, n)
            points = np.append(points, new_points, axis=0)
            # remember the biggest triangle that was inside, if later
            # sample some additional points need to be added
            if t.within(self.polygon) and t.area > t_area:
                big_t = [t][0]
                t_area = t.area
        points = self._check_enough_points_sampled(n, points, big_t)
        points = super()._cut_points(n, points)
        return super().divide_points_to_space_variables(points)

    def _sample_in_triangulation(self, t, n):
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        corners = np.array([[x0, y0], [x1, y1], [x2, y2]])
        scaled_n = int(np.ceil(t.area/self.polygon.area * n))
        new_points = self._random_points_in_triangle(scaled_n, corners)
        # when the polygon has holes or is non convex, it can happen
        # that the triangle is not completly in the polygon
        if not t.within(self.polygon):
            inside = self.is_inside(new_points)
            index = np.where(inside)[0]
            new_points = new_points[index]
        return new_points

    def _random_points_in_triangle(self, n, corners):
        bary_coords = np.random.uniform(0, 1, (n, 2))
        # if a barycentric coordinates is bigger then 1, mirror them at the
        # point (0.5, 0.5). Stays uniform.
        index = np.where(bary_coords.sum(axis=1) > 1)[0]
        bary_coords[index] = np.subtract([1, 1], bary_coords[index])
        axis_1 = np.multiply(corners[1]-corners[0], bary_coords[:, :1])
        axis_2 = np.multiply(corners[2]-corners[0], bary_coords[:, 1:])
        return np.add(np.add(corners[0], axis_1), axis_2).astype(np.float32)

    def _check_enough_points_sampled(self, n, points, big_t):
        # if not enough points are sampled, create some new points in the biggest
        # triangle
        while len(points) < n:
            points = np.append(points,
                               self._sample_in_triangulation(big_t, n-len(points)),
                               axis=0)
        return points

    def sample_grid(self, n):
        points = self._create_points_in_bounding_box(n)
        points = self._delete_outside(points)
        points = super()._check_grid_enough_points(n, points)
        points = super()._cut_points(n, points)
        return super().divide_points_to_space_variables(points)

    def _create_points_in_bounding_box(self, n):
        bounds = self.bounding_box()
        side_lengths = [bounds[1]-bounds[0], bounds[3]-bounds[2]]
        scaled_n = int(n * side_lengths[0]*side_lengths[1]/self.polygon.area)
        vertices_of_box = np.array([[bounds[0], bounds[2]], [bounds[1], bounds[2]],
                                    [bounds[1], bounds[3]], [bounds[0], bounds[3]]])
        return Parallelogram.grid_in_box(self, scaled_n, side_lengths, vertices_of_box)

    def _delete_outside(self, points):
        inside = self.is_inside(points)
        index = np.where(np.invert(inside))[0]
        return np.delete(points, index, axis=0)

    @property
    def boundary(self):
        return BoundaryDomain2D(self)


class BoundaryDomain2D(BoundaryDomain):
    """Handels the boundary in 2D.
    """

    def __init__(self, domain: Domain2D):
        super().__init__(domain)
        self.domain = domain
        self.normal_list = None

    def is_inside(self, points):
        points = super()._check_single_point(points)
        points = super().return_space_variables_to_point_list(points)
        on_bound = np.empty(len(points), dtype=bool)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            distance = self.domain.polygon.boundary.distance(point)
            on_bound[i] = (np.abs(distance) <= self.tol)
        return on_bound.reshape(-1, 1)

    def bounding_box(self):
        return self.domain.bounding_box()

    def outline(self):
        return self.domain.outline()

    def sample_random_uniform(self, n):
        line_points = np.random.uniform(0, self.domain.polygon.boundary.length, n)
        return self._transform_points_to_boundary(n, np.sort(line_points))

    def sample_grid(self, n):
        line_points = np.linspace(0, self.domain.polygon.boundary.length, n+1)[:-1]
        return self._transform_points_to_boundary(n, line_points)

    def _transform_points_to_boundary(self, n, line_points):
        """Transform points that lay between 0 and polygon.boundary.length to 
        the surface of this polygon. The points have to be ordered from smallest
        to biggest.
        """
        outline = self.domain.outline()
        index = 0
        current_length = 0
        points = np.zeros((n, 2))
        for boundary_part in outline:
            points, index, current_length = \
                self._distribute_line_to_boundary(points, index, line_points,
                                                  boundary_part, current_length)
        return super().divide_points_to_space_variables(points)

    def _distribute_line_to_boundary(self, points, index, line_points,
                                     corners, current_length):
        corner_index = 0
        side_length = np.linalg.norm(corners[1] - corners[0])
        while index < len(line_points):
            if line_points[index] <= current_length + side_length:
                point = self._translate_point_to_bondary(index, line_points,
                                                         corners, current_length,
                                                         corner_index, side_length)
                points[index] = point
                index += 1
            else:
                corner_index += 1
                current_length += side_length
                if corner_index >= len(corners) - 1:
                    break
                side_length = np.linalg.norm(corners[corner_index+1]
                                             - corners[corner_index])
        return points, index, current_length

    def _translate_point_to_bondary(self, index, line_points, corners,
                                    current_length, corner_index, side_length):
        coord = line_points[index] - current_length
        new_point = (corners[corner_index] + coord/side_length *
                     (corners[corner_index+1] - corners[corner_index]))
        return new_point

    def normal(self, points):
        points = super().return_space_variables_to_point_list(points)
        outline = self.domain.outline()
        self._compute_normals(outline)
        index = self._where_on_boundary(points, outline)
        return self.normal_list[index].astype(np.float32)

    def _compute_normals(self, outline):
        if self.normal_list is not None:
            return
        face_number = sum([len(corners) for corners in outline])
        self.normal_list = np.zeros((face_number, 2))
        index = 0
        for corners in outline:
            for i in range(len(corners)-1):
                normal = self._compute_local_normal_vector(corners[i+1], corners[i])
                self.normal_list[index] = normal
                index += 1

    def _compute_local_normal_vector(self, end_corner, start_corner):
        conect_vector = end_corner - start_corner
        side_length = np.linalg.norm(conect_vector)
        normal = conect_vector[::-1] / side_length
        normal[1] *= -1
        return normal

    def _where_on_boundary(self, points, outline):
        points = super()._check_single_point(points)
        index = -1 * np.ones(len(points), dtype=int)
        counter = 0
        for corners in outline:
            for i in range(len(corners)-1):
                line = s_geo.LineString([corners[i], corners[i+1]])
                not_found = np.where(index < 0)[0]
                for k in not_found:
                    point = s_geo.Point(points[k])
                    distance = line.distance(point)
                    if np.abs(distance) <= self.tol:
                        index[k] = counter
                    else:
                        break
                counter += 1
        return index


class Circle(Domain2D):
    """Class for circles.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the circle, e.g. center = [5,0].
    radius : number or callable
        The radius of the circle.
    resolution : number, optional
        The resolution that should be used to approximate the circle with a polygon.
        The number of used vertices is around 4*resolution.
        See shapely for exact informations.
    """
    def __new__(cls, space, center, radius, resolution=10, tol=1e-06):
        if callable(center) or callable(radius):
            params = {'center': center, 'radius': radius, 'resolution': resolution}
            return LambdaDomain(class_=cls, params=params, space=space, dim=2, tol=tol)
        return super(Circle, cls).__new__(cls)

    def __init__(self, space, center, radius, resolution=10, tol=1e-06):
        circle = s_geo.Point(center).buffer(radius, resolution=resolution)
        super().__init__(polygon=circle, space=space, tol=tol)


class Ellipse(Domain2D):
    """Class for ellipse.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    center : array_like or callable
        The center of the circle, e.g. center = [5,0].
    radius_x, radius_y : number or callable
        The 'radius' of the ellipse in each axis-direction.
    angle: number or callable
        The angle between the cart. x-axis and the 'x-axis' of the ellipse.
    resolution : number, optional
        The number of vertices that should be used to approixmate the ellipse.
        The number of used vertices is around 4*resolution.
        See shapely for exact informations.
    """
    def __new__(cls, space, center, radius_x, radius_y, angle=0,
                resolution=10, tol=1e-06):
        if any([callable(center), callable(radius_x),
                callable(radius_y), callable(angle)]):
            params = {'center': center, 'radius_x': radius_x, 'radius_y': radius_y,
                      'angle': angle, 'resolution': resolution}
            return LambdaDomain(class_=cls, params=params, space=space, dim=2, tol=tol)
        return super(Ellipse, cls).__new__(cls)

    def __init__(self, space, center, radius_x, radius_y, angle=0,
                 resolution=10, tol=1e-06):
        circle = s_geo.Point(center).buffer(1, resolution=resolution)
        ellipse = shapely.affinity.scale(circle, radius_x, radius_y)
        ellipse = shapely.affinity.rotate(ellipse, -angle)
        super().__init__(polygon=ellipse, space=space, tol=tol)


class Parallelogram(Domain2D):
    """Class for arbitrary parallelograms, even if time dependet
    will always stay a parallelogram.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    corner_dl, corner_dr, corner_tl : array_like or callable
        Three corners of the parallelogram, in the following order
        |       tl ----- x
        |      /        /
        |     /        /
        |    dl ----- dr
        (dl = down left, dr = down right, tl = top left)
        E.g. for the unit square: corner_dl = [0,0], corner_dr = [1,0],
                                  corner_tl = [0,1].
    """
    def __new__(cls, space, corner_dl, corner_dr, corner_tl, tol=1e-06):
        if any([callable(corner_dl), callable(corner_dr), callable(corner_tl)]):
            params = {'corner_dl': corner_dl, 'corner_dr': corner_dr,
                      'corner_tl': corner_tl}
            return LambdaDomain(class_=cls, params=params, space=space, dim=2, tol=tol)
        return super(Parallelogram, cls).__new__(cls)

    def __init__(self, space, corner_dl, corner_dr, corner_tl, tol=1e-06):
        vertices = self._get_vertices(corner_dl, corner_dr, corner_tl)
        super().__init__(polygon=s_geo.Polygon(vertices), space=space, tol=tol)

    def _get_vertices(self, corner_dl, corner_dr, corner_tl):
        # cast to array:
        corner_dl = np.array(corner_dl)
        corner_dr = np.array(corner_dr)
        corner_tl = np.array(corner_tl)
        return [corner_dl, corner_dr, corner_dr+(corner_tl-corner_dl), corner_tl]

    def sample_grid(self, n):
        vertices = np.array(self.polygon.exterior.coords)
        side_lengths = self._get_side_lengths(vertices)
        points = self.grid_in_box(n, side_lengths, vertices)
        points = super()._check_grid_enough_points(n, points)
        return super().divide_points_to_space_variables(points)

    def _get_side_lengths(self, vertices):
        side_x = np.linalg.norm(vertices[3]-vertices[0])
        side_y = np.linalg.norm(vertices[3]-vertices[0])
        return [side_x, side_y]

    def grid_in_box(self, n, side_lengths, vertices):
        """ Samples grid points inside the parallelogram.
        """
        nx = int(np.sqrt(n*side_lengths[0]/side_lengths[1]))
        ny = int(np.sqrt(n*side_lengths[1]/side_lengths[0]))
        x = np.linspace(0, 1, nx+2)[1:-1]
        y = np.linspace(0, 1, ny+2)[1:-1]
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        trans_matrix = np.column_stack((vertices[3]-vertices[0],
                                        vertices[1]-vertices[0]))
        points = [np.matmul(trans_matrix, p) for p in points]
        points = np.add(points, vertices[0])
        return points


class Polygon(Domain2D):
    """Class for polygons.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    vertices : list of lists or callable, optional 
        The corners/vertices of the polygon. Can be eihter in clockwise or counter-
        clockwise order. 
    shapely_polygon : shapely.geometry.Polygon, optional
        Instead of defining the corner points, it is also possible to give a already
        existing shapely.Polygon object.  
    """
    def __new__(cls, space, vertices=None, shapely_poly=None, tol=1e-06):
        if vertices is not None and isinstance(vertices, (list, np.ndarray)):
            if callable(vertices):
                params = {'vertices': vertices}
                return LambdaDomain(class_=cls, params=params, space=space,
                                    dim=2, tol=tol)
        return super(Polygon, cls).__new__(cls)

    def __init__(self, space, vertices=None, shapely_poly=None, tol=1e-06):
        if isinstance(shapely_poly, s_geo.Polygon):
            super().__init__(polygon=shapely_poly, space=space, tol=tol)
        if vertices is not None:
            super().__init__(polygon=s_geo.Polygon(vertices), space=space, tol=tol)
        else:
            raise ValueError('Needs either vertices to create a new'
                             + ' polygon, or a existing shapely polygon.')
