import torch
from .space import Space

class Points():
    """
    A set of points in a space, stored as a torch.Tensor.
    Supports indexing and slicing as []
    """
    def __init__(self, data, space, **kwargs):
        self.data = torch.as_tensor(data, **kwargs)
        self.space = space
        assert len(self.data.shape) == 2
        assert self.data.shape[1] == self.space.dim
    
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
            out[var] = self.data[:,self._compute_var_slice(var)]
        return out

    @property
    def as_tensor(self):
        return self.data
    
    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return "Points:\n{}".format(self.coordinates)
    
    def _compute_var_slice(self, var):
        start = 0
        for v in self.variables:
            if v == var:
                stop = start + self.space[v]
                break
            else:
                start += self.space[v]
        return slice(start, stop, None)
    
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
        out = self.data[val[0]]
        
        if len(val) == 2:
            if isinstance(val[1], str):
                if val[1] in self.variables:
                    out = out[self._compute_var_slice(val[1])]
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
            yield Point(self.data[i,:], self.space)
    
    def __add__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self.data + other.data, self.space)

    def __sub__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self.data - other.data, self.space)

    def __mul__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self.data * other.data, self.space)

    def __pow__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self.data ** other.data, self.space)

    def __truediv__(self, other):
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self.data / other.data, self.space)
    



class Point(Points):
    def __init__(self, data, space, **kwargs):
        data = torch.as_tensor(data, **kwargs)
        if len(data.shape) == 1:
            super().__init__(data.unsqueeze(0), space)
        elif len(data.shape) == 2:
            assert data.shape[0] == 1
            super().__init__(data, space)