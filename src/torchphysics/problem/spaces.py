from collections import Counter
import torch

class Space(Counter):

    def __init__(self, variables_dims):
        # set counter of variable names and their dimensionalities
        super().__init__(variables_dims)

    def __mul__(self, other):
        assert isinstance(other, Space)
        return Space(self + other)

    @property
    def dim(self):
        return sum(self.values())
    

    def as_variable_dict(self, points):
        """Divides sample points of the form torch.tensor(number_of_points, self.dim)
        to a dict containing each variable of this Space.

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
        if isinstance(points, dict):
            assert points.keys() == self.keys()
            return points
        output = {}
        current_dim = 0
        for vname in self:
            v_dim = self[vname]
            output[vname] = points[:, current_dim:current_dim+v_dim].astype(torch.float32)
            current_dim += v_dim
        return output

    def as_tensor(self, point_dic):
        """Concatenates sample points from a dict back to the form 
        torch.tensor(number_of_points, self.dim)

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
        if isinstance(point_dic, (list, torch.tensor)):
            return point_dic
        point_list = list(point_dic.values())
        return torch.column_stack(point_list)


class R1(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 1})


class R2(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 2})


class R3(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 3})