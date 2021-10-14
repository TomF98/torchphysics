'''File contains different helper functions to get specific informations about
the computed solution.
'''
import time
import torch
import numpy as np


def get_min_max_inside(model, solution_name, point_sampler,
                       device='cpu', requieres_grad=False):
    '''Computes the minimum and maximum values of the model w.r.t. the given
    variables.

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which values should be computed.
    solution_name : str
        The output function for which the min. and max. should be computed.
    point_sampler : DataSampler
        A sampler that creates the points where the model should be evaluated.
    device : str or torch device
        The device of the model.    

    Returns
    -------
    float
        The minimum of the model.
    float 
        The maximum  of the model.
    '''
    print('-- Start evaluation of minimum and maximum --')
    input_dic = _create_input(point_sampler, device, requieres_grad)
    start = time.time()
    pred = model(input_dic)
    end = time.time()
    pred = pred[solution_name].data.cpu().numpy()
    max_pred = np.max(pred)
    min_pred = np.min(pred)
    print('Time to evaluate model:', end - start)
    print('Found the values')
    print('Max:', max_pred)
    print('Min:', min_pred)
    return min_pred, max_pred

def _create_input(point_sampler, device, track_grad):
    input_dic = point_sampler.sample_points()
    for vname in input_dic:
        if not isinstance(input_dic[vname], torch.Tensor):
            input_dic[vname] = torch.tensor(input_dic[vname], device=device, 
                                            requires_grad=track_grad)
    return input_dic