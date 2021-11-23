import abc
import torch

from ...utils.user_fun import UserFunction


class Domain:

    def __init__(self, space, dim=None):
        self.space = space
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim
        self._user_volume = None

    def set_necessary_variables(self, *domain_params):
        # create a set of variables/spaces that this domain needs to be properly defined
        self.necessary_variables = set()
        for d_param in domain_params:
            for k in d_param.necessary_args:
                self.necessary_variables.add(k)
        assert not any(var in self.necessary_variables for var in self.space)

    def transform_to_user_functions(self, *domain_params):
        out = []
        for d_param in domain_params:
            if not isinstance(d_param, UserFunction):
                d_param = UserFunction(d_param)
            out.append(d_param)
        return tuple(out)

    @property
    def boundary(self):
        """
        The boundary domain. Is not implemented for some domains, as for
        example boundarys of other domains.
        """
        raise NotImplementedError

    def set_volume(self, volume):
        self._user_volume = UserFunction(volume)

    @abc.abstractmethod
    def _get_volume(self, **params):
        raise NotImplementedError

    def volume(self, **params):
        if self._user_volume is None:
            return self._get_volume(**params)
        else:
            return self._user_volume()

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
        from .domainoperations.union import UnionDomain
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
        from .domainoperations.cut import CutDomain
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
        from .domainoperations.intersection import IntersectionDomain
        return IntersectionDomain(self, other)

    def __mul__(self, other):
        """Creates the cartesian product of this domain and another domain.

        Parameters
        ----------
        other : Domain
            The other domain to create the cartesian product with.
            Should lie in a disjoint space.
        """
        from .domainoperations.product import ProductDomain
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
    def _contains(self, points, **params):
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

    def __call__(self, **data):
        raise NotImplementedError

    def get_num_of_params(self, **params):
        # finds the number of params, for which points should be sampled
        num_of_params = 1
        if len(params) > 0:
            num_of_params = len(list(params.values())[0])
        return num_of_params

    def compute_n_from_density(self, d, **params):
        volume = self.volume(**params)
        if len(volume) > 1:
            raise ValueError(f"""Sampling with a density is only possible for one
                                given pair of parameters. Found {len(volume)} 
                                different pairs. If sampling with a density is needed, 
                                a loop should be used.""")
        n = torch.ceil(volume / d**self.dim)
        return int(n)

    def _repeat_params(self, n, **args):
        repeated_params = {}
        param_len = 1
        for key, domain_param in args.items():
            repeated_params[key] = torch.repeat_interleave(domain_param, n, dim=0)
            param_len = len(repeated_params[key])
        if param_len > 1:
            n = 1
        return n, repeated_params


class BoundaryDomain(Domain):
    
    def __init__(self, domain):
        assert isinstance(domain, Domain)
        super().__init__(space=domain.space, dim=domain.dim-1)
        self.domain = domain
        self.necessary_variables = self.domain.necessary_variables

    def __call__(self, **data):
        evaluate_domain = self.domain(**data)
        return evaluate_domain.boundary

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
