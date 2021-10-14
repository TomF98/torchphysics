from collections import Counter


class Space(Counter):

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

    @property
    def dim(self):
        return sum(self.values())


class R1(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 1})


class R2(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 2})


class R3(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 3})
