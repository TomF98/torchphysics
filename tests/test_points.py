import torch
import torchphysics as tp

def test_points_get_item():
    X = tp.spaces.R2('x')
    T = tp.spaces.R1('t')
    p = tp.spaces.Points(torch.tensor([[2.0,1.0,2.5],
                                       [1.0,3.0,1.0],
                                       [1.0,1.0,4.0],
                                       [1.0,5.0,1.0],
                                       [6.0,1.0,1.0]]),
                         X*T)

    assert p[1,'x'] == torch.tensor([1.0, 3.0])
    assert p[1:3,'x'] == torch.tensor([[1.0, 3.0],
                                       [1.0, 1.0],
                                       [1.0, 5.0]])
    assert p[2:5,:] == p[2:5]
    assert p[2:5,:] == tp.spaces.Points(torch.tensor([[1., 1., 4.],
                                                      [1., 5., 1.],
                                                      [6., 1., 1.]]),
                                        X*T)
    assert p[2:5,:].coordinates == {
        'x': torch.tensor([[1., 1.],
                           [1., 5.],
                           [6., 1.]]),
        't': torch.tensor([[4.],
                           [1.],
                           [1.]])}

def test_points_torch_fun():
    assert
