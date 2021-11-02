import numpy as np
import trimesh
import logging

from .domain import Domain, BoundaryDomain
from .domain2D import Polygon


class Domain3D(Domain):

    def __init__(self, space, mesh: trimesh.Trimesh, tol=0.000001):
        super().__init__(space, dim=3, tol=tol)
        self.mesh = mesh
        # Trimesh gives a warning when not enough points are sampled. We already
        # take care of this problem. So set the logging only to errors.
        logging.getLogger("trimesh").setLevel(logging.ERROR)

    def is_inside(self, points):
        points = super().return_space_variables_to_point_list(points)
        return self.mesh.contains(points).reshape(-1,1)

    @property
    def boundary(self):
        return BoundaryDomain3D(self)

    def bounding_box(self):
        bound_corners = self.mesh.bounds
        return bound_corners.T.flatten()

    def export_file(self, name_of_file):
        '''Exports the mesh to a file.

        Parameters
        ----------
        name_of_file : str
            The name of the new file.
        '''
        self.mesh.export(name_of_file)

    def slice_with_plane(self, space, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
        '''Slices the mesh with the given plane.

        Parameters
        ----------
        plane_origin : array_like, optional
            The origin of the plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough to give
            the wanted direction ofthe  normal vector, it does not need to
            have norm = 1. 

        Returns
        ----------
        Polygon
            A Polygon object from domain2D that represents the slice of the
            original mesh with the plane.
        '''
        # slice the mesh
        #rotation_matrix = self._create_inverse_rotation_matrix(plane_normal)
        #self.mesh.apply_translation(plane_origin)
        #self.mesh.apply_transform(rotation_matrix)
        plane_normal = self._normalize_vector(plane_normal)
        slice = self.mesh.section(plane_origin=plane_origin,
                                  plane_normal=plane_normal)
        if slice is None:
            raise ValueError('slice of mesh and plane is empty!')
        # create the 2d object
        slice_2D = slice.to_planar(to_2D=np.eye(4), check=False)[0]
        polygon = slice_2D.polygons_full[0]
        polygon = polygon.simplify(self.tol)
        return Polygon(space=space, vertices=polygon, tol=self.tol)

    def sample_random_uniform(self, n):
        points = np.empty((0,self.dim))
        while len(points) < n:
            new_points = trimesh.sample.volume_mesh(self.mesh, n-len(points))
            points = np.append(points, new_points, axis=0)
        return super().divide_points_to_space_variables(points)

    def __add__(self, other):
        assert other.dim == 3
        new_mesh = trimesh.boolean.union([self, other])
        return Domain3D(space=self.space, mesh=new_mesh,
                        tol=np.min[self.tol, other.tol])

    def __sub__(self, other):
        assert other.dim == 3
        new_mesh = trimesh.boolean.difference([self, other])
        return Domain3D(space=self.space, mesh=new_mesh,
                        tol=np.min[self.tol, other.tol])

    def __and__(self, other):
        assert other.dim == 3
        new_mesh = trimesh.boolean.intersection([self, other])
        return Domain3D(space=self.space, mesh=new_mesh,
                        tol=np.min[self.tol, other.tol])

    def _create_inverse_rotation_matrix(self, orientation):
        orientation = self._normalize_vector(orientation)
        m = np.linalg.inv(self._create_rotation_matrix(orientation))
        matrix = np.eye(4)
        matrix[:3, :3] = m
        return matrix

    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1):
            vector /= norm
        return vector

    def _create_rotation_matrix(self, orientation):
        # create a matrix to rotate the domain, to be parallel to the z-axis
        # From:
        # https://math.stackexchange.com/questions
        # /180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        c = -orientation[2] # cosine of angle between orientation and (0,0,-1)
        v = [-orientation[1], orientation[0], 0] # cross pord. 
        if c == -1: # orientation is (0,0,1) -> do nothing
            return np.eye(3)
        else: 
            I = np.eye(3)
            M = np.array([[0, 0, v[1]], [0, 0, -v[0]], [-v[1], v[0], 0]])
            R = I + M + 1/(1+c)*np.linalg.matrix_power(M, 2)
            return R


class BoundaryDomain3D(BoundaryDomain):
    """Handels the boundary in 3D.
    """
    def __init__(self, domain: Domain3D):
        super().__init__(domain)
        self.domain = domain
        self.normal_list = self.domain.mesh.face_normals

    def is_inside(self, points):
        distance = trimesh.proximity.signed_distance(self.domain.mesh, points)
        abs_dist = np.absolute(distance)
        on_bound = (abs_dist <= self.tol)
        return on_bound.reshape(-1,1)

    def bounding_box(self):
        return self.domain.bounding_box()

    def sample_random_uniform(self, n):
        points = trimesh.sample.sample_surface(self.domain.mesh, n)[0]
        return super().divide_points_to_space_variables(points)

    def sample_grid(self, n):
        points = trimesh.sample.sample_surface_even(self.domain.mesh, n)[0]
        points = super()._check_grid_enough_points(n, points)
        return super().divide_points_to_space_variables(points)

    def normal(self, points):
        points = super().return_space_variables_to_point_list(points)
        index = self.domain.mesh.nearest.on_surface(points)[2]
        normals = self.normal_list[index]
        return normals.astype(np.float32)


class Box(Domain3D):
    """Class for arbitrary boxes in 3D.

    Parameters
    ----------
    corner_o, corner_x, corner_y, corner_z : list, array or callable
        Four corners of the Box, in the following form
            |      . ----- .
            |     / |     /|
            |   z ----- .  |
            |   |   |   |  |
            |   |  y ---|--.
            |   | /     | /
            |   o ----- x
        (o = corner origin, x = corner in "x" direction,
         y = corner in "y" direction, z = corner in "z" direction)
        Can also be rotated, if needed. 
        E.g.: corner_oc = [0,0,0], corner_xc = [2,0,0],
              corner_yc = [0,1,0], corner_zc = [0,0,3].
    """
    def __new__(cls, space, corner_o, corner_x, corner_y, corner_z, tol=1e-06):
        if any([callable(corner_o), callable(corner_x), 
                callable(corner_y), callable(corner_z)]):
                params = {'corner_o': corner_o, 'corner_x': corner_x, 
                          'corner_y': corner_y, 'corner_z': corner_z}
                return LambdaDomain(class_=cls, params=params, space=space,
                                    dim=3, tol=tol)
        return super(Box, cls).__new__(cls)

    def __init__(self, space, corner_o, corner_x, corner_y, corner_z, tol=1e-06):
        mesh = self._create_unit_cube()
        trans_matrix = self._create_transform_matrix(corner_o, corner_x, 
                                                     corner_y, corner_z)
        mesh.apply_transform(trans_matrix)
        mesh.apply_translation(corner_o)
        super().__init__(space=space, mesh=mesh, tol=tol)

    def _create_unit_cube(self):
        cube = trimesh.primitives.Box()
        cube.apply_translation([0.5, 0.5, 0.5])
        return cube

    def _create_transform_matrix(self, corner_o, corner_x, corner_y, corner_z):
        matrix = np.eye(4)
        matrix[:3, 0] = np.array(corner_x) - np.array(corner_o)
        matrix[:3, 1] = np.array(corner_y) - np.array(corner_o)
        matrix[:3, 2] = np.array(corner_z) - np.array(corner_o)
        return matrix

    def sample_random_uniform(self, n):
        points = self.mesh.sample_volume(n)
        return super().divide_points_to_space_variables(points)


class Sphere(Domain3D):
    """Class for arbitrary spheres.

    Parameters
    ----------
    center : list, array or callable
        The center of the sphere, e.g. center = [5,0,1].
    radius : number or callable
        The radius of the sphere.
    """
    def __new__(cls, space, center, radius, tol=1e-06):
        if any([callable(center), callable(radius)]):
                params = {'center': center, 'radius': radius}
                return LambdaDomain(class_=cls, params=params, space=space,
                                    dim=3, tol=tol)
        return super(Sphere, cls).__new__(cls)

    def __init__(self, space, center, radius, tol=1e-06):
        mesh = trimesh.primitives.Sphere(center=center, radius=radius)
        super().__init__(space=space, mesh=mesh, tol=tol)


class Cylinder(Domain3D):
    """ Class for arbitrary cylinders.

    Parameters
    ----------
    center : list, array or callable
        The center of the cylinder, e.g. center = [5,0,1].
    radius : number or callable
        The radius of the cylinder.
    height : number or callable
        The total height of the cylinder.
    orientation : list, array or callable
        The orientation of the cylinder. A vector that is orthogonal to the circle
        areas of the cylinder.
    """
    def __new__(cls, space, center, radius, height, orientation=[0, 0, 1], tol=1e-06):
        if any([callable(center), callable(radius), 
                callable(height), callable(orientation)]):
                params = {'center': center, 'radius': radius, 
                          'height': height, 'orientation': orientation}
                return LambdaDomain(class_=cls, params=params, space=space,
                                    dim=3, tol=tol)
        return super(Cylinder, cls).__new__(cls)

    def __init__(self, space, center, radius, height, orientation=[0, 0, 1],
                 tol=1e-06):
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        rotation = super()._create_inverse_rotation_matrix(orientation)
        mesh.apply_transform(rotation)
        mesh.apply_translation(center)
        super().__init__(space=space, mesh=mesh, tol=tol)


class Capsule(Domain3D):
    """ Class for arbitrary capsules. (Cylinders with a half-sphere
    on each side)

    Parameters
    ----------
    center : list, array or callable
        The center of the capsule, e.g. center = [5,0,1].
    radius : number or callable
        The radius of the capsule.
    height : number or callable
        The total height of the capsule.
    orientation : list, array or callable
        The orientation of the capsule. A vector that is orthogonal to the circle
        areas of the capsule.
    """
    def __new__(cls, space, center, radius, height, orientation=[0, 0, 1], tol=1e-06):
        if any([callable(center), callable(radius), 
                callable(height), callable(orientation)]):
                params = {'center': center, 'radius': radius, 
                          'height': height, 'orientation': orientation}
                return LambdaDomain(class_=cls, params=params, space=space,
                                    dim=3, tol=tol)
        return super(Capsule, cls).__new__(cls)

    def __init__(self, space, center, radius, height, orientation=[0, 0, 1],
                 tol=1e-06):
        mesh = trimesh.primitives.Capsule(radius=radius, height=height)
        rotation = super()._create_inverse_rotation_matrix(orientation)
        mesh.apply_transform(rotation)
        mesh.apply_translation(center)
        super().__init__(space=space, mesh=mesh, tol=tol)



class Polyhedron(Domain):
    '''Class for 3D polyhedron.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    vertices : list of lists or callable, optional 
        The vertices of the polyhedron.
    faces : list of lists, optional 
        A list that contains which vetrices have to be connected to create the faces
        of the polyhedron. If for example the vertices 1, 2 and 3 have should be 
        connected do: faces = [[1,2,3]]
    file_name : str or file-like object, optional
        A data source to load a existing polyhedron/mesh.
    file_type : str, optional
        The file type, e.g. 'stl'. See trimesh.available_formats() for all supported
        file types.
    '''
    def __new__(cls, space, vertices=None, faces=None, file_name=None,
                 file_type='stl', tol=1e-06):
        if callable(vertices):
            params = {'vertices': vertices}
            return LambdaDomain(class_=cls, params=params, space=space,
                                dim=3, tol=tol)
        return super(Polyhedron, cls).__new__(cls)

    def __init__(self, space, vertices=None, faces=None, file_name=None,
                 file_type='stl', tol=1e-06):
        if vertices is not None and faces is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.fix_normals()
        elif file_name is not None:
            self.mesh = trimesh.load_mesh(file_name, file_type=file_type)
        else:
            raise ValueError('Needs either vertices and faces to create a new' \
                             'polygon, or a file to load a existing one.')
        super().__init__(space=space, mesh=mesh, tol=tol)

"""
class Polygon3D(Domain):
    '''Class for polygons in 3D.

    Parameters
    ----------
    vertices : list of lists, optional 
        The vertices of the polygon.
    faces : list of lists, optional 
        A list that contains which vetrices have to be connected to create the faces
        of the polygon. If for example the vertices 1, 2 and 3 have should be 
        connected do: faces = [[1,2,3]]
    file_name : str or file-like object, optional
        A data source to load a existing polygon/mesh.
    file_type : str, optional
        The file type, e.g. 'stl'. See trimesh.available_formats() for all supported
        file types.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary
    '''
    def __init__(self, vertices=None, faces=None, file_name=None, file_type='stl',
                 tol=1e-06):
        if vertices is not None and faces is not None:
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh.fix_normals()
        elif file_name is not None:
            self.mesh = trimesh.load_mesh(file_name, file_type=file_type)
        else:
            raise ValueError('Needs either vertices and faces to create a new' \
                             'polygon, or a file to load a existing one.')
        super().__init__(dim=3, volume=self.mesh.volume, 
                         surface=sum(self.mesh.area_faces), tol=tol)
        # Trimesh gives a warning when not enough points are sampled. We already
        # take care of this problem. So set the logging only to errors.
        logging.getLogger("trimesh").setLevel(logging.ERROR)

    def export_file(self, name_of_file):
        '''Exports the mesh to a file.

        Parameters
        ----------
        name_of_file : str
            The name of the file.
        '''
        self.mesh.export(name_of_file)

    def project_on_plane(self, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
        '''Projects the polygon on a plane. 

        Parameters
        ----------
        plane_origin : array_like, optional
            The origin of the projection plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 

        Returns
        ----------
        Polygon2D
            The polygon that is the outline of the projected original mesh on 
            the plane.
        '''
        norm = np.linalg.norm(plane_normal)
        if not np.isclose(norm, 1):
            plane_normal /= norm
        polygon = trimesh.path.polygons.projected(self.mesh, origin=plane_origin,
                                                  normal=plane_normal)
        polygon = polygon.simplify(self.tol)
        return Polygon2D(shapely_polygon=polygon, tol=self.tol)

    def slice_with_plane(self, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
        '''Slices the polygon with a plane.

        Parameters
        ----------
        plane_origin : array_like, optional
            The origin of the plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 

        Returns
        ----------
        Polygon2D
            The polygon that is the outline of the projected original mesh on 
            the plane.
        '''
        norm = np.linalg.norm(plane_normal)
        if not np.isclose(norm, 1):
            plane_normal /= norm
        rotaion_matrix = self._create_rotation_matrix_to_plane(plane_normal)
        slice = self.mesh.section(plane_origin=plane_origin,
                                  plane_normal=plane_normal)
        if slice is None:
            raise ValueError('slice of mesh and plane is empty!')
        slice_2D = slice.to_planar(to_2D=rotaion_matrix, check=False)[0]
        polygon = slice_2D.polygons_full[0]
        polygon = polygon.simplify(self.tol)
        return Polygon2D(shapely_polygon=polygon, tol=self.tol)

    def _create_rotation_matrix_to_plane(self, plane_normal):
        u = [plane_normal[1], -plane_normal[0], 0]
        cos = plane_normal[2]
        sin = np.sqrt(plane_normal[0]**2 + plane_normal[1]**2)
        matrix = [[cos+u[0]**2*(1-cos), u[0]*u[1]*(1-cos),    -u[1]*sin, 0], 
                  [u[0]*u[1]*(1-cos),   cos+u[1]**2*(1-cos),  u[0]*sin,  0], 
                  [-u[1]*sin,           u[0]*sin,             -cos,      0],
                  [0,                   0,                    0,         1]]
        return matrix

    def is_inside(self, points):
        '''Checks if the given points are inside the mesh.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside,
            or false if not.
        '''
        return self.mesh.contains(points).reshape(-1,1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the surface of the mesh.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        distance = trimesh.proximity.signed_distance(self.mesh, points)
        abs_dist = np.absolute(distance)
        on_bound = (abs_dist <= self.tol)
        return on_bound.reshape(-1,1)

    def _random_sampling_inside(self, n):
        points = np.empty((0,self.dim))
        missing = n
        while len(points) < n:
            new_points = trimesh.sample.volume_mesh(self.mesh, missing)
            points = np.append(points, new_points, axis=0)
            missing -= len(new_points)
        return points.astype(np.float32)

    def _grid_sampling_inside(self, n):
        raise NotImplementedError #Needs 3D Box class

    def _random_sampling_boundary(self, n):
        return trimesh.sample.sample_surface(self.mesh, n)[0].astype(np.float32)

    def _grid_sampling_boundary(self, n):
        points = trimesh.sample.sample_surface_even(self.mesh, n)[0]
        points = super()._check_boundary_grid_enough_points(n, points)
        return points.astype(np.float32)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector has to be computed,e.g.
            [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        if not all(self.is_on_boundary(points)):
            print('Warning: some points are not at the boundary!')
        index = self.mesh.nearest.on_surface(points)[2]
        mesh_normals = self.mesh.face_normals
        normals = np.zeros((len(points), self.dim))
        for i in range(len(points)):
            normals[i, :] = mesh_normals[index[i]]
        return normals.astype(np.float32)
"""