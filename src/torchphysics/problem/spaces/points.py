import torch
from .space import Space

class Points():
    """
    A set of points in a space, stored as a torch.Tensor.
    Supports indexing and slicing as []
    """
    def __init__(self, data, space, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.space = space
        assert len(self._t.shape) == 2
        assert self._t.shape[1] == self.space.dim
    
    @classmethod
    def from_coordinates(cls, coords):
        """Concatenates sample coordinates from a dict to create a point
        object.

        Parameters
        ----------
        coords : dict
            The dictionary containing the data for every variable.

        Returns
        -------
        points : Points or Point
            the created points object
        """
        point_list = []
        space = {}
        n = coords[list(coords.keys())[0]].shape[0]
        for vname in coords:
            coords[vname] = torch.as_tensor(coords[vname])
            assert coords[vname].shape[0] == n
            point_list.append(coords[vname])
            space[vname] = coords[vname].shape[1]
        if n == 1:
            return Point(torch.column_stack(point_list), Space(space))
        else:
            return Points(torch.column_stack(point_list), Space(space))
    
    @property
    def dim(self):
        return self.space.dim
    
    @property
    def variables(self):
        return self.space.variables
    
    @property
    def coordinates(self):
        """
        Returns a dict containing the coordinates of all points for each
        variable, e.g. {'x': torch.Tensor, 't': torch.Tensor}
        """
        out = {}
        for var in self.space:
            out[var] = self._t[:,self._variable_slices[var]]
        return out
    
    @property
    def _variable_slices(self):
        start = 0
        slices = {}
        for v in self.space:
            stop = start + self.space[v]
            slices[v] = slice(start, stop, None)
            start += self.space[v]
        return slices

    @property
    def as_tensor(self):
        return self._t
    
    def __len__(self):
        return self._t.shape[0]

    def __repr__(self):
        return "{}:\n{}".format(self.__class__.__name__, self.coordinates)
    
    def __getitem__(self, val):
        """
        Supports operations like points[1:3,'x']. If a variable is given,
        this will return a torch.Tensor with the data. If not, it will return
        a new, sliced, point.

        Notes
        -----
        This operation does not support slicing single dimensions from a
        variable directly, however, this can be done on the output.
        """
        val = tuple(val)
        out = self._t[val[0],:]

        if len(val) == 2:
            if isinstance(val[1], str):
                if val[1] in self.variables:
                    if isinstance(val[0], slice):
                        out = out[:, self._variable_slices[val[1]]]
                    elif isinstance(val[0], int):
                        out = out[self._variable_slices[val[1]]]
                    return out
                raise KeyError(f"Variable {val[1]} does not exist.")
            elif isinstance(val[1], slice):
                if val[1] != slice(None, None, None):
                    raise TypeError("Slicing is not supported for variable-dimension.")
            else:
                raise TypeError("Wrong type for indexing or slicing.")
        if len(val) > 2:
            raise IndexError("too many indices for Points (max. 3 indices).")

        if isinstance(val[0], slice):
            return Points(out, self.space)
        elif isinstance(val[0], int):
            return Point(out, self.space)
        else:
            raise TypeError(f"{val[0]} should be slice or int.")

    
    def __iter__(self):
        """
        Iterates through points. It is in general not recommended
        to use this operation because it may lead to huge (and therefore
        slow) loops.
        """
        for i in range(len(self)):
            yield Point(self._t[i,:], self.space)

    def __eq__(self, other):
        return self.space == other.space and self._t == other._t
    
    def __add__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t + other._t, self.space)

    def __sub__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t - other._t, self.space)

    def __mul__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t * other._t, self.space)

    def __pow__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t ** other._t, self.space)

    def __truediv__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t / other._t, self.space)
    
    def __or__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(torch.cat([self._t, other._t], dim=0), self.space)
    
    def join(self, other):
        assert isinstance(other, Points)
        assert self.space.keys().isdisjoint(other.space)
        return Points(torch.cat([self._t, other._t], dim=1), self.space * other.space)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        A helper method to create compatibility with most torch operations.
        Heavily inspired by the official torch documentation.
        """
        if kwargs is None:
            kwargs = {}
        args_list = [a._t if hasattr(a, '_t') else a for a in args]
        spaces = tuple(a.space for a in args if hasattr(a, 'space'))
        assert len(spaces) > 0
        assert all(space == spaces[0] for space in spaces)
        ret = func(*args_list, **kwargs)
        return Points(ret, spaces[0])


class Point(Points):
    def __init__(self, data, space, **kwargs):
        data = torch.as_tensor(data, **kwargs)
        if len(data.shape) == 1:
            super().__init__(data.unsqueeze(0), space)
        elif len(data.shape) == 2:
            assert data.shape[0] == 1
            super().__init__(data, space)
