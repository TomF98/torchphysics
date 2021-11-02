import abc
import torch
import copy 

from ...utils.user_fun import UserFunction


class Domain:

    def __init__(self, constructor, params, space, dim=None):
        self.space = space
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim

        self.constructor = constructor
        self.params = params

        # create a set of variables/spaces that this domain needs to be properly defined
        self.necessary_variables = set()
        for key in self.params:
            if callable(self.params[key]):
                self.params[key] = UserFunction(params[key])
                for k in self.params[key].necessary_args:
                    self.necessary_variables.add(k)
        assert not any(var in self.necessary_variables for var in self.space)

    @property
    def boundary(self):
        # Domain object of the boundary
        raise NotImplementedError

    @property
    def inner(self):
        # open domain
        raise NotImplementedError

    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        if self.space != other.space:
            raise ValueError("""united domains should lie in the same space.""")
        from .newdomainoperations import UnionDomain
        return UnionDomain(self, other)

    def __sub__(self, other):
        """Creates the cut of domain other from self.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        if self.space != other.space:
            raise ValueError("""complemented domains should lie in the same space.""")
        from .newdomainoperations import CutDomain
        return CutDomain(self, other)

    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to lie in the same space.
        """
        if self.space != other.space:
            raise ValueError("""Intersected domains should lie in the same space.""")
        from .newdomainoperations import IntersectionDomain
        return IntersectionDomain(self, other)

    def __mul__(self, other):
        """Creates the cartesian product of this domain and another domain.

        Parameters
        ----------
        other : Domain
            The other domain to create the cartesian product with.
            Should lie in a disjoint space.
        """
        from .newdomainoperations import ProductDomain
        return ProductDomain(self, other)

    def __contains__(self, points):
        """Checks for every point in points if it lays inside the domain.

        Parameters
        ----------
        points : list or array
            The list of diffrent or a single point that should be checked.
            E.g in 2D: points = [[2, 4], [9, 6], ....]

        Returns
        -------
        array
            A an array of the shape (len(points), 1) where every entry contains
            true if the point was inside or false if not.
        """
        return self._contains(points)

    @abc.abstractmethod
    def contains(self, points, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def bounding_box(self, **params):
        """Computes the bounds of the domain.

        Returns 
        list :
            A list with the length of 2*self.dim.
            It has the form [axis_1_min, axis_1_max, axis_2_min, axis_2_max, ...], 
            where min and max are the minimum and maximum value that the domain
            reaches in each dimension-axis.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n=None, d=None, **params):
        """Greates a equdistant grid in the domain.

        Parameters
        ----------
        n : int
            The number of points that should be created.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n=None, d=None, **params):
        """Greates a random uniform points in the domain.

        Parameters
        ----------
        n : int
            The number of points that should be created.
        """
        raise NotImplementedError

    def _divide_points_to_space_variables(self, points):
        """Divides sample points of the form np.array(number_of_points, self.dim)
        to each variable of the given Space.

        Parameters
        ----------
        points: list, array
            The created sample/data points, need to fit the given dimension

        Returns
        -------
        dict
            A dictionary containing the input points but split up, to each 
            variable. E.g Space = R1('x')*R1('y') then the output would be
            output = {'x': points[:, 0:1], 'y': points[:, 1:2]}
        """
        output = {}
        current_dim = 0
        for vname in self.space:
            v_dim = self.space[vname]
            output[vname] = points[:, current_dim:current_dim+v_dim]
            current_dim += v_dim
        return output

    def _return_space_variables_to_point_list(self, point_dic):
        """Concatenates sample points from a dict back to the form 
        np.array(number_of_points, self.dim)

        Parameters
        ----------
        point_dic: dic
            The dictionary of points 
            (most likely created with divide_points_to_space_variables)

        Returns
        -------
        points: array
            the point array of the form np.array(number_of_points, self.dim)
        """
        # if the points are not a dictonary just return
        # (not created with our sampling)
        if isinstance(point_dic, (list, torch.Tensor)):
            return point_dic
        point_list = []
        for vname in self.space:
            point_list.append(point_dic[vname])
        return torch.column_stack(point_list)

    def __call__(self, **data):
        """
        (Partially) evaluate given lambda functions.
        """
        evaluated_params = {}
        for key in self.params:
            evaluated_params[key] = self._call_param(self.params[key], data)
        if all(var in data for var in self.necessary_variables):
            return self.constructor(space=self.space, **evaluated_params)
        else:
            self.params = evaluated_params
            return self

    def _call_param(self, param, args):
        if callable(param):
            if all(arg in args for arg in param.necessary_args):
                return param(**args)
            else:
                # to avoid manipulation of given param obj, we create a copy
                copy.deepcopy(param).set_default(**args)
        return param

    def _domain_construction(self, **args):
        # first find number of parameters:
        args_len = 1
        if len(args) > 0:
            args_len = len(list(args.values())[0])
        d_params = {'param_len': args_len}
        # evaluate possible domain values that are dependent on the input args
        for key, domain_param in self.params.items():
            if callable(domain_param):
                d_params[key] = domain_param(**args)[:, None] 
            else:
                if isinstance(domain_param, torch.Tensor):
                    d_params[key] = domain_param
                else: 
                    d_params[key] = torch.tensor(domain_param).float()
        return d_params


class BoundaryDomain(Domain):
    
    def __init__(self, domain):
        assert isinstance(domain, Domain)
        super().__init__(space=domain.space, dim=domain.dim-1,
                         constructor=None, params={})
        self.domain = domain

    def __call__(self, **data):
        evaluate_domain = self.domain(**data)
        return evaluate_domain.boundary

    def _domain_construction(self, **params):
        return self.domain._domain_construction(**params)

    def bounding_box(self, **params):
        return self.domain.bounding_box(**params)

    @abc.abstractmethod
    def normal(self, points, **params):
        """Computes the normal vector at each point in points.

        Parameters
        ----------
        points : list or array
            A list of diffrent or a single point for which the normal vector 
            should be computed. The points must lay on the boundary of the domain.
            E.g in 2D: points = [[2, 4], [9, 6], ....]        

        Returns
        -------
        array
            The array is of the shape (len(points), self.dim) and contains the 
            normal vector at each entry from points.
        """
        pass
