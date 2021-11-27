from collections import Counter, OrderedDict
from typing import Iterable

import torch


class Space(Counter, OrderedDict):

    def __init__(self, variables_dims):
        # set counter of variable names and their dimensionalities
        super().__init__(variables_dims)

    def __mul__(self, other):
        assert isinstance(other, Space)
        return Space(self + other)

    def __contains__(self, space):
        if isinstance(space, str):
            return super().__contains__(space)
        if isinstance(space, Space):
            return (self & space) == space
        else:
            raise TypeError
    
    def __getitem__(self, val):
        if isinstance(val, slice):
            keys = list(self.keys())
            new_slice = slice(keys.index(val.start) if val.start is not None else None,
                              keys.index(val.stop) if val.stop is not None else None,
                              val.step)
            new_keys = keys[new_slice]
            return Space({k: self[k] for k in new_keys})
        if isinstance(val, list) or isinstance(val, tuple):
            return Space({k: self[k] for k in val})
        else:
            return super().__getitem__(val)

    @property
    def dim(self):
        return sum(self.values())
    
    @property
    def variables(self):
        """
        A unordered (!) set of variables.
        """
        return set(self.keys())


    """
    Python recipe (see official Python docs) to maintain the insertion order.
    This way, dimensions with identical variable names will be joined, all
    other dimensions will be kept in the order of their creation by products
    or __init__.
    """
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, dict(OrderedDict(self)))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class R1(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 1})


class R2(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 2})


class R3(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 3})
